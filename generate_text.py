import os
import sys
import argparse
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        low_cpu_mem_usage=True if args.torch_version >= 1.9 else False
    )


    if device == "cuda":
        model.half()
        model = model.cuda()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    def evaluate(
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        yield output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.95, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=50, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Evaluate Pruned Model",
    ).queue().launch(share=args.share_gradio, inbrowser=True,server_name="0.0.0.0",server_port=7860)  # server_name="0.0.0.0", share=args.share_gradio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--share_gradio', action='store_true')

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    # args.share_gradio = True  # 本地调式用

    main(args)