'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import csv
import os
import random
import sys
from utils.get_calibration_samples import get_examples
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM

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

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('using ddp...')
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if args.base_model == 'Vicuna_7B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            trust_remote_code=True, device_map=device_map, use_cache=False,
        )
        config_path = '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/'
    else:
        sys.exit(0)

    print(model)

    def count_parameters(model):
        # Sum up all parameters
        return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(model)
    print(f"Total number of parameters before pruning: {total_params}")

    if not os.path.exists(args.output_dir + '{}/{}/'.format(args.pruning_method, args.base_model)):
        os.mkdir(args.output_dir + '{}/{}/'.format(args.pruning_method, args.base_model))


    if args.pruning_method == 'taylor':  # ref to Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
        # model.half()
        model = model.cuda()
        result_csv_weight = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "weight_score.csv")
        result_csv_block = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_all.csv")
        result_csv_block_detail = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_detail.csv")
        result_csv_block_sort = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_sorted.csv")
        block_order_path = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_order.csv")

        print("Do forward to collect gradient information")
        salience_dict = {}
        example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=128).to(args.device)

        for i in range(0, example_prompts.size(0), args.batch_size):
            example_prompts_tmp = example_prompts[i: i + args.batch_size]
            loss = model(example_prompts_tmp, labels=example_prompts_tmp).loss
            loss.backward()
            for k, param in model.named_parameters():
                if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                    salience = param * param.grad
                    salience = salience.data.clone().float()

                    if k not in salience_dict.keys():
                        salience_dict[k] = salience
                    else:
                        salience_dict[k] += salience

            model.zero_grad()

        # Compute scores of weight matrices -> Collec them
        block_info = {}
        with open(result_csv_weight, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["weight_name", "weight_score"])
            for k, param in model.named_parameters():
                if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                    block_idx = ".".join(k.split(".")[:3])  # 'model.layers.i'
                    if "proj" in k or "lm_head" in k:  # output_dim x input_dim
                        weight_imp = (
                            salience_dict[k].abs().pow(args.norm_power).sum(1)
                        )  # [output_dim]
                    elif "norm" in k:  # [output_dim]
                        weight_imp = salience_dict[k].abs().pow(args.norm_power)

                    if args.weight_reduction == "sum":
                        weight_imp = weight_imp.sum(dim=0)
                    elif args.weight_reduction == "mean":
                        weight_imp = weight_imp.mean(dim=0)
                    elif args.weight_reduction == "max":
                        weight_imp = weight_imp.max(dim=0)[0]
                    elif args.weight_reduction == "prod":
                        weight_imp = torch.prod(weight_imp, dim=0)
                    else:
                        raise NotImplementedError

                    weight_imp = weight_imp.item()
                    logwriter.writerow([k, weight_imp])
                    print([k, weight_imp])
                    if block_idx not in block_info.keys():
                        block_info[block_idx] = [weight_imp]
                    else:
                        block_info[block_idx].append(weight_imp)

        # Compute block-level importance
        block_info_summary = {}
        with open(result_csv_block, "w") as logfile, open(
                result_csv_block_detail, "w"
        ) as logfile_detail:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["block_name", "block_score"])
            logwriter_detail = csv.writer(logfile_detail, delimiter=",")
            logwriter_detail.writerow(["block_name", "all_weight_scores"])
            for k, v in block_info.items():
                print(k, v)
                logwriter_detail.writerow([k] + v)

                block_imp = torch.tensor(v)
                if args.block_reduction == "sum":
                    block_imp = block_imp.sum(dim=0)
                elif args.block_reduction == "mean":
                    block_imp = block_imp.mean(dim=0)
                elif args.block_reduction == "max":
                    block_imp = block_imp.max(dim=0)[0]
                elif args.block_reduction == "prod":
                    block_imp = torch.prod(block_imp, dim=0)
                else:
                    raise NotImplementedError

                block_imp = block_imp.item()
                logwriter.writerow([k, block_imp])
                block_info_summary[k] = block_imp

        for k in ["model.norm.weight", "lm_head.weight"]:
            if k in block_info_summary:
                del block_info_summary[k]
        sorted_items = sorted(block_info_summary.items(), key=lambda x: x[1])
        block_order = []
        with open(result_csv_block_sort, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["rank", "block_name", "block_score", "block_index"])
            for rank, (key, value) in enumerate(sorted_items, start=1):
                logwriter.writerow([rank, key, value, key.split(".")[-1]])
                print([rank, key, value, key.split(".")[-1]])
                block_order.append(int(key.split(".")[-1]))

        with open(block_order_path, "w") as logfile_order:
            logwriter_order = csv.writer(logfile_order, delimiter=",")
            logwriter_order.writerow(block_order)

        print(f"=== block order removed: {block_order_path}")
        print(block_order)
        print(f"len: {len(block_order)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--output_dir', type=str,
                        default="/public/ly/SBF/pruning_method/",
                        help='output directory')
    parser.add_argument('--pruning_method', type=str, default="magnitude_l1", help='pruning name')

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    # pruning
    parser.add_argument("--norm_power", type=int, default=1, help="1 or 2 for l-p norm")
    parser.add_argument(
        "--weight_reduction", type=str, default="sum", help="sum, mean, max, prod"
    )
    parser.add_argument(
        "--block_reduction", type=str, default="sum", help="sum, mean, max, prod"
    )
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    ## CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python pruning_taylor_pall.py --base_model Vicuna_7B  --pruning_method taylor --output_dir ~
    ## CUDA_VISIBLE_DEVICES=1,2 TRANSFORMERS_OFFLINE=1 python pruning_method.py --base_model llama3-8b  --pruning_method magnitude_l2  --norm_power 2
    ## CUDA_VISIBLE_DEVICES=1,2 TRANSFORMERS_OFFLINE=1 python pruning_method.py --base_model llama3-8b  --pruning_method ppl
    # CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 lm_eval --model hf  --model_args pretrained=/public/MountData/yaolu/,trust_remote_code=True  --tasks arc_easy  --device cuda:0  --batch_size auto  --num_fewshot 0
    main(args)
