'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import gc
import os
import random
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_list_to_txt(my_list, file_path):
    try:
        # Open the file at the specified path in write mode
        with open(file_path, 'w') as file:
            # Iterate over each item in the list
            for item in my_list:
                # Write each item to the file, followed by a newline character
                file.write(f"{item}\n")
        print(f"List has been saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main(args):
    # Load Pruned Model
    set_random_seed(args.seed)

    device_map = "balanced_low_0"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('using ddp...')
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if args.base_model == 'llama3-8b':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/LLAMA3_8B/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/LLAMA3_8B/',
            trust_remote_code=True, use_cache=False, device_map=device_map
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/LLAMA3_8B/'
    elif args.base_model == 'Vicuna_7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/'
    elif args.base_model == 'Qwen1.5-7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/Qwen1.5-7B/models--Qwen--Qwen1.5-7B/'
    elif args.base_model == 'Gemma2-2b':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/gemma-2-2b-it/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/gemma-2-2b-it/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/gemma-2-2b-it/'
    elif args.base_model == 'chatglm2-6b':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/chatglm2-6b/models--THUDM--chatglm2-6b/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/chatglm2-6b/models--THUDM--chatglm2-6b/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/chatglm2-6b/models--THUDM--chatglm2-6b/'
    elif args.base_model == 'Llama-3.1-8B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/',
            torch_dtype=torch.bfloat16,
            device_map="balanced_low_0", # auto
            use_cache=False,
            cache_dir='/public/MountData/yaolu/LLM_pretrained/Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/'
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/'
    else:
        sys.exit(0)

    print(model)

    # #### new layer = L1 + L2  #####
    # if args.fusion:
    #     name = [1,2,3,4,5,(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),18,19,20,21,22,23,24,25,(26,27),28,(29,30),31]
    #     # name = [i for i in range(32)]  # for test
    #
    #     new_config = AutoConfig.from_pretrained(config_path, num_hidden_layers=len(name), trust_remote_code=True)
    #     new_model = AutoModelForCausalLM.from_config(new_config)
    #
    #     # 复制参数
    #     new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
    #     new_model.model.norm.load_state_dict(model.model.norm.state_dict())
    #     new_model.lm_head.load_state_dict(model.lm_head.state_dict())
    #
    #     # 处理层
    #     for i, idx in enumerate(name):
    #         if isinstance(idx, tuple):
    #             layers_to_fuse = idx
    #             fused_state_dict = model.model.layers[layers_to_fuse[0]].state_dict()
    #             for layer_idx in layers_to_fuse[1:]:
    #                 layer_state_dict = model.model.layers[layer_idx].state_dict()
    #                 for key in fused_state_dict.keys():
    #                     fused_state_dict[key] += layer_state_dict[key]  # 直接加
    #             new_model.model.layers[i].load_state_dict(fused_state_dict)
    #         else:
    #             layer_state_dict = model.model.layers[idx].state_dict()
    #             new_model.model.layers[i].load_state_dict(layer_state_dict)

    # # #### new layer = a * L1 + (1-a) * L2  #####
    # name = [1,2,3,4,5,(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),18,19,20,21,22,23,24,25,(26,27),28,(29,30),31]
    # # name = [i for i in range(32)]  # for test
    #
    # new_config = AutoConfig.from_pretrained(config_path, num_hidden_layers=len(name), trust_remote_code=True)
    # new_model = AutoModelForCausalLM.from_config(new_config)
    #
    # # 复制参数
    # new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
    # new_model.model.norm.load_state_dict(model.model.norm.state_dict())
    # new_model.lm_head.load_state_dict(model.lm_head.state_dict())
    #
    # a = 0.5
    #
    # # 处理层
    # for i, idx in enumerate(name):
    #     if isinstance(idx, tuple):
    #         layers_to_fuse = idx
    #         fused_state_dict = model.model.layers[layers_to_fuse[0]].state_dict()
    #         for layer_idx in layers_to_fuse[1:]:
    #             layer_state_dict = model.model.layers[layer_idx].state_dict()
    #             for key in fused_state_dict.keys():
    #                 fused_state_dict[key] = a * fused_state_dict[key] + (1-a) * layer_state_dict[key]  #
    #         new_model.model.layers[i].load_state_dict(fused_state_dict)
    #     else:
    #         layer_state_dict = model.model.layers[idx].state_dict()
    #         new_model.model.layers[i].load_state_dict(layer_state_dict)


    # ###### selective pruning ######
    def remove_elements(lst, elements_to_remove):
        return list(filter(lambda element: element not in elements_to_remove, lst))

    def random_select_and_sort(lst, num_elements):
        if num_elements > len(lst):
            raise ValueError("num_elements is larger than the size of the list")

        selected_elements = random.sample(lst, num_elements)
        selected_elements.sort()

        return selected_elements

    if args.base_model == 'Gemma2-2b':
        layer_num = [i for i in range(26)]
        remain_num = 20
    else:
        layer_num = [i for i in range(32)]
        remain_num = 24

    if args.pr_method == 'random':
        remained = random_select_and_sort(layer_num, remain_num)
    elif args.pr_method == 'tail':
        remained = [i for i in range(remain_num)]
    else:
        num = args.remove_layer
        if args.base_model == 'Llama-3.1-8B-Instruct':
            block = [26, 25, 24, 28, 27, 23, 29, 22, 20, 21, 19, 18, 30, 17, 13, 16, 14, 15, 12, 10, 11, 9, 8, 7, 31, 6, 5, 0, 4, 2, 3, 1]  # llama3.1 taylor
        if args.base_model == 'Vicuna_7B':
            block = [0, 1, 29, 28, 30, 26, 27, 25, 24, 23, 21, 22, 31, 19, 12, 20, 18, 13, 14, 11, 17, 8, 10, 9, 16, 7, 15, 2, 6, 5, 3, 4]

        block_list = block[:num]
        block_list = sorted(block_list)
        removed = block_list
        print(removed)  # Gemma2-2b BI
        remained = remove_elements(layer_num, removed)

    # removed = [8,9,10,11,12,24,25,26]  # ppl_llama3_8b
    # removed = [1, 3, 4, 9, 10, 11, 12, 14]  # magnitude_l1_Qwen1.5-7B
    # removed = [1, 3, 4, 5, 6, 7, 9, 10]  # magnitude_l2_Qwen1.5-7B
    # removed = [0, 1, 2, 3, 6, 7, 8, 11]  # magnitude_l1_Vicuna_7B
    # removed = [0, 1, 3, 6, 7, 8, 9, 11]  # magnitude_l2_Vicuna_7B
    # removed = [28, 29, 30, 31]  # Vicuna_7B
    # removed = [1, 2, 3, 4, 28, 29, 30, 31]  # Vicuna_7B

    print('remained: {}'.format(remained))
    new_config = AutoConfig.from_pretrained(config_path, num_hidden_layers=len(remained), trust_remote_code=True)
    new_model = AutoModelForCausalLM.from_config(new_config)
    # 复制参数
    new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
    new_model.model.norm.load_state_dict(model.model.norm.state_dict())
    new_model.lm_head.load_state_dict(model.lm_head.state_dict())
    for i in range(len(remained)):
        layer_state_dict = model.model.layers[remained[i]].state_dict()
        new_model.model.layers[i].load_state_dict(layer_state_dict)

    new_model.to(args.device)
    print(new_model)

    if args.save_model:
        output_lora_dir = '/public/MountData/yaolu/LLM_pretrained/pruned_model/oneshot/pruned_{}_{}_{}/'.format(args.base_model, args.pr_method, args.remove_layer)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        new_model.save_pretrained(output_lora_dir)
        tokenizer.save_pretrained(output_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--output_dir', type=str,
                        default="/public/MountData/yaolu/LLM_pretrained/pruned_model/lora-alpaca-llama/",
                        help='output directory')
    parser.add_argument('--pr_method', type=str, default="ppl", help='device')
    parser.add_argument('--remove_layer', type=int, default=16, help='batch size')

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    parser.add_argument('--fusion', action='store_true', help='if merge model')

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    ## CUDA_VISIBLE_DEVICES=2,3 TRANSFORMERS_OFFLINE=1 python prune_llm.py --base_model Llama-3.1-8B-Instruct --save_model  --pr_method taylor --remove_layer 1
    # CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 lm_eval --model hf  --model_args pretrained=/public/MountData/yaolu/,trust_remote_code=True  --tasks arc_easy  --device cuda:0  --batch_size auto  --num_fewshot 0
    main(args)
