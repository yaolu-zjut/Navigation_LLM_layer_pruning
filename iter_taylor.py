import argparse
import os
import random
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import csv

from utils.get_calibration_samples import get_examples
device = "cuda" if torch.cuda.is_available() else "cpu"

def remove_elements(lst, elements_to_remove):
    return list(filter(lambda element: element not in elements_to_remove, lst))


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


def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
    nsamples = test_ids.numel() // seq_len

    test_set = []
    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_set.append({
            'input_ids': batch,
            'labels': batch
        })
    return test_set


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.prune_model_path,
        trust_remote_code=True,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.prune_model_path,
        use_fast=False,
        trust_remote_code=True
    )

    lora_model = PeftModel.from_pretrained(
        model,
        args.lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    model = model.cuda()
    for name, param in model.named_parameters():
        param.requires_grad = True

    result_csv_weight = os.path.join(args.output_dir + '{}/{}/'.format(args.pr_method, args.base_model),
                                     "weight_score.csv")
    result_csv_block = os.path.join(args.output_dir + '{}/{}/'.format(args.pr_method, args.base_model),
                                    "block_score_all.csv")
    result_csv_block_detail = os.path.join(args.output_dir + '{}/{}/'.format(args.pr_method, args.base_model),
                                           "block_score_detail.csv")
    result_csv_block_sort = os.path.join(args.output_dir + '{}/{}/'.format(args.pr_method, args.base_model),
                                         "block_score_sorted.csv")
    block_order_path = os.path.join(args.output_dir + '{}/{}/'.format(args.pr_method, args.base_model),
                                    "block_order.csv")

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

    # Compute scores of weight matrices -> Collect them
    block_info = {}
    with open(result_csv_weight, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(["weight_name", "weight_score"])
        for k, param in model.named_parameters():
            if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                block_idx = ".".join(k.split(".")[:3])  # 'model.layers.i'
                if "proj" in k or "lm_head" in k:  # output_dim x input_dim
                    weight_imp = salience_dict[k].abs().pow(args.norm_power).sum(1)  # [output_dim]
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
    with open(result_csv_block, "w") as logfile, open(result_csv_block_detail, "w") as logfile_detail:
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

    blocks = block_order[:1]
    blocks = sorted(blocks)
    print(blocks)

    layer_num = len(block_order)
    removed = blocks
    print(f"Removed layers: {removed}")
    remained = remove_elements(list(range(layer_num)), removed)
    print(f"Remained layers: {remained}")

    new_config = AutoConfig.from_pretrained(args.prune_model_path, num_hidden_layers=len(remained),
                                            trust_remote_code=True)
    new_model = AutoModelForCausalLM.from_config(new_config)

    new_model.model.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
    new_model.model.norm.load_state_dict(model.model.norm.state_dict())
    new_model.lm_head.load_state_dict(model.lm_head.state_dict())
    for j in range(len(remained)):
        layer_state_dict = model.model.layers[remained[j]].state_dict()
        new_model.model.layers[j].load_state_dict(layer_state_dict)

    new_model.to(args.device)
    print(new_model)

    if args.save_model:
        output_lora_dir = '/public/MountData/yaolu/LLM_pretrained/pruned_model/Iterative_lora/Iterative_pruned_{}_taylor{}_to_{}{}/'.format(
            args.base_model, args.remove_layer, args.pr_method, args.remove_layer + 1)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        new_model.save_pretrained(output_lora_dir)
        tokenizer.save_pretrained(output_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM')

    # Model Type&Path
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--lora_path', type=str, help='lora name')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--base_model', type=str, default="Gemma2-2b", help='base model name')
    parser.add_argument('--pr_method', type=str, default="taylor", help='device')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    parser.add_argument('--remove_layer', type=int, default=1, help='batch size')
    parser.add_argument('--output_dir', type=str,
                        default="/public/ly/SBF/pruning_method/iter/",
                        help='output directory')
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
main(args)
