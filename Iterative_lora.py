import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
                trust_remote_code=True, device_map='auto'
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.prune_model_path,
        use_fast=False, trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(
            model, args.lora_path,
            torch_dtype=torch.float16,
        )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    remain_num = 14
    remained = [i for i in range(remain_num)]

    print('remained: {}'.format(remained))
    new_config = AutoConfig.from_pretrained(args.prune_model_path, num_hidden_layers=len(remained), trust_remote_code=True)
    new_model = AutoModelForCausalLM.from_config(new_config)
    # 复制参数
    new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
    new_model.model.norm.load_state_dict(model.model.norm.state_dict())
    new_model.lm_head.load_state_dict(model.lm_head.state_dict())
    for i in range(len(remained)):
        layer_state_dict = model.model.layers[remained[i]].state_dict()
        new_model.model.layers[i].load_state_dict(layer_state_dict)

    print(new_model)

    if args.save_model:
        output_lora_dir = '/public/MountData/yaolu/LLM_pretrained/pruned_model/Iterative_pruned_{}_tail8to{}/'.format(args.base_model, args.pr_method)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        new_model.save_pretrained(output_lora_dir)
        tokenizer.save_pretrained(output_lora_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM')

    # Model Type&Path
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--lora_path', type=str, help='lora name')
    parser.add_argument('--base_model', type=str, default="Gemma2-2b", help='base model name')
    parser.add_argument('--pr_method', type=str, default="tail16", help='device')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    main(args)