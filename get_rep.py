import argparse
import torch
import sys

from peft import PeftModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils.CKA import linear_CKA
from utils.get_calibration_samples import get_examples

'''GhatGLM 暂时没解决'''


def get_hidden_states(model, inputs):
    hidden_states = []

    def hook(module, input, output):
        hidden_states.append(output)

    hooks = []
    if args.model == 'ChatGLM6B':
        for layer in model.transformer.layers:  # all
            hook_handle = layer.register_forward_hook(hook)
            hooks.append(hook_handle)
    else:
        for layer in model.model.layers:  # all
            hook_handle = layer.register_forward_hook(hook)
            hooks.append(hook_handle)

    with torch.no_grad():
        if args.model == 'ChatGLM6B':
            _ = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        else:
            _ = model(inputs, labels=inputs)

    for hook_handle in hooks:
        hook_handle.remove()

    return hidden_states



def main(args):
    if args.model == 'llama2_7b':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/LLAMA2_7B/', use_fast=False, trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
                '/public/MountData/yaolu/LLM_pretrained/LLAMA2_7B/', trust_remote_code=True,
                low_cpu_mem_usage=True if args.torch_version >=1.9 else False
        )
    elif args.model == 'BaiChuan7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Baichuan_7B/', use_fast=False, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Baichuan_7B/', trust_remote_code=True,
            low_cpu_mem_usage=True if args.torch_version >= 1.9 else False
        )
    elif args.model == 'llama3-8b':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/LLAMA3_8B/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/LLAMA3_8B/',
            trust_remote_code=True, use_cache=False
        )
    elif args.model == 'Vicuna_7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            trust_remote_code=True, use_cache=False
        )
    elif args.model == 'Qwen1.5-7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            trust_remote_code=True, use_cache=False
        )
    elif args.model == 'ChatGLM6B':  # have some problem
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/chatglm6b/', use_fast=False, trust_remote_code=True
        )
        model = AutoModel.from_pretrained("/public/MountData/yaolu/LLM_pretrained/chatglm6b/", trust_remote_code=True)
    elif args.model == 'lora':
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     trust_remote_code=True, device_map='auto'
                                                     )
        lora_model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.float16,
        )
        model = lora_model.merge_and_unload()
    else:
        print('Not support {}!'.format(args.model))
        sys.exit(0)

    model.to(args.device)
    print(model)
    prompts = get_examples('bookcorpus', tokenizer, 3 * args.num_examples, seq_len=128).to(args.device)

    print("Start Backwarding in iterative steps...")

    for kk in range(3):
        example_prompts = prompts[kk * args.num_examples:(kk + 1) * args.num_examples, :]
        print(example_prompts.shape)
        hidden_states = get_hidden_states(model, example_prompts)
        b, h, w = hidden_states[0][0].shape[0], hidden_states[0][0].shape[1], hidden_states[0][0].shape[2]
        print(hidden_states[0][0].shape)  # torch.Size([batch, 64, 4096])

        sim_matrix = torch.zeros((len(hidden_states), len(hidden_states)))
        for i in range(len(hidden_states)):
            for j in range(len(hidden_states)):
                if i >= j:
                    sim_matrix[i,j] = sim_matrix[j,i] = linear_CKA(hidden_states[i][0].view(b, h * w).cuda(), hidden_states[j][0].view(b, h * w).cuda())

        torch.save(sim_matrix, '/public/ly/SBF/img/sim_matrix_{}_id{}_batch{}.pth'.format(args.model, kk, args.num_examples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating Similarity')
    parser.add_argument('--num_examples', type=int, default=64)
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    parser.add_argument('--model', type=str, default="llama3-8b", help='device')
    parser.add_argument('--model_path', type=str, help='prune model name')
    parser.add_argument('--lora_path', type=str, help='lora name')
    args = parser.parse_args()

    args.torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    main(args)