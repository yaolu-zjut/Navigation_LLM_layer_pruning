'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import csv
import gc
import os
import random
import sys

from torch import nn

from utils.get_calibration_samples import get_examples

sys.path.append('/public/ly/SBF/evaluate/')
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM
from ppldataset import get_wikitext2, get_ptb, process_data
from utils.eval import eval_zero_shot

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


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric


def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def main(args):
    # Load Pruned Model
    set_random_seed(args.seed)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('using ddp...')
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if args.base_model == 'llama3-8b':
        tokenizer = AutoTokenizer.from_pretrained(
            'LLAMA3_8B/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            'LLAMA3_8B/',
            trust_remote_code=True, use_cache=False, device_map=device_map, low_cpu_mem_usage=True if args.torch_version >=1.9 else False
        )
        config_path = 'LLAMA3_8B/'
    elif args.base_model == 'Vicuna_7B':
        tokenizer = AutoTokenizer.from_pretrained(
            'Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            'Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
            trust_remote_code=True, device_map=device_map, use_cache=False,
        )
        config_path = 'Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/'
    elif args.base_model == 'Qwen1.5-7B':
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            use_fast=False, trust_remote_code=True # delete this for BI, add torch_dtype=torch.bfloat16 for taylor
        )
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen1.5-7B/models--Qwen--Qwen1.5-7B/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = 'Qwen1.5-7B/models--Qwen--Qwen1.5-7B/'
    elif args.base_model == 'Gemma2-2b':
        tokenizer = AutoTokenizer.from_pretrained(
            'gemma-2-2b-it/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            'gemma-2-2b-it/',
            trust_remote_code=True, device_map=device_map, use_cache=False
        )
        config_path = 'gemma-2-2b-it/'
    elif args.base_model == 'Llama-3.1-8B-Instruct':  # NEED TO CHECK
        tokenizer = AutoTokenizer.from_pretrained(
            'Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/',
            use_fast=False, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            'Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/',
            torch_dtype=torch.bfloat16,  # delete this for BI, maintain for taylor
            device_map=device_map,
            use_cache=False,
            cache_dir='Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/'
        )
        config_path = 'Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/'
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

    if args.pruning_method == 'magnitude_l1':  # ref to Pruning filters for efficient convnets
        if args.norm_power != 1:
            print('Norm_power should be set to 1')
            sys.exit(0)

        result_csv_weight = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "weight_score.csv")
        result_csv_block = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_all.csv")
        result_csv_block_detail = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_detail.csv")
        result_csv_block_sort = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_sorted.csv")
        block_order_path = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_order.csv")

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
                            param.data.clone().float().abs().pow(args.norm_power).sum(1)
                        )  # [output_dim]
                    elif "norm" in k:  # [output_dim]
                        weight_imp = param.data.clone().float().abs().pow(args.norm_power)

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

    elif args.pruning_method == 'magnitude_l2':  # ref to Pruning filters for efficient convnets
        if args.norm_power != 2:
            print('Norm_power should be set to 2')
            sys.exit(0)

        result_csv_weight = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "weight_score.csv")
        result_csv_block = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_all.csv")
        result_csv_block_detail = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_detail.csv")
        result_csv_block_sort = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_score_sorted.csv")
        block_order_path = os.path.join(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model), "block_order.csv")

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
                            param.data.clone().float().abs().pow(args.norm_power).sum(1)
                        )  # [output_dim]
                    elif "norm" in k:  # [output_dim]
                        weight_imp = param.data.clone().float().abs().pow(args.norm_power)

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

    elif args.pruning_method == 'ppl':  # ref to Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
        # model.half()
        ppl = PPLMetric(model.cuda(), tokenizer, ['wikitext2'], 128, device=args.device)
        print("PPL before pruning: {}".format(ppl))

        def remove_elements(lst, elements_to_remove):
            return list(filter(lambda element: element not in elements_to_remove, lst))

        ppl_list = []

        layer_num = [i for i in range(32)]
        for i in range(len(layer_num)):
            remained = remove_elements(layer_num, [i])
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

            total_params = count_parameters(new_model)
            print(f"Total number of parameters after pruning: {total_params}")

            ppl = PPLMetric(new_model.to(args.device), tokenizer, ['wikitext2'], 128, device=args.device, batch_size=1)
            del new_model
            print("PPL after pruning: {}".format(ppl))
            ppl_list.append(ppl)

        torch.save(ppl_list, '/public/ly/SBF/pruning_method/ppl/wikitext2_{}.pth'.format(args.base_model))

    elif args.pruning_method == 'taylor':  # ref to Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
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

    elif args.pruning_method == 'BI':  # ref to ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
        # model.half()  # will result in NAN  maybe add lm_head feat
        if not os.path.exists(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model)):
            os.mkdir(args.output_dir+'{}/{}/'.format(args.pruning_method, args.base_model))

        def calculate_layer_importance(model, inputs):
            # Get all intermediate outputs
            layer_inputs = []

            # Hook to capture input to each layer
            def hook(module, input, output):
                layer_inputs.append(input[0])  # Save input to layer

            # Register hook on each layer
            hooks = []
            for layer in model.model.layers:
                hook_handle = layer.register_forward_hook(hook)
                hooks.append(hook_handle)

            # Run forward pass
            print(inputs.is_cuda)
            model(inputs, labels=inputs)

            # Calculate the importance for each layer
            layer_importance = []
            for i in range(len(layer_inputs) - 1):
                X_i = layer_inputs[i].detach()
                X_i1 = layer_inputs[i + 1].detach()

                # Calculate the dot product
                dot_product = torch.sum(X_i * X_i1)

                # Calculate norms
                norm_X_i = torch.norm(X_i)
                norm_X_i1 = torch.norm(X_i1)

                # Calculate importance
                importance = 1 - (dot_product / (norm_X_i * norm_X_i1))
                layer_importance.append(importance.item())

            # Remove hooks
            for hook_handle in hooks:
                hook_handle.remove()

            return layer_importance

        example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=128).to(args.device)  # better to use 10, 5

        # Calculate layer importance
        importance_scores = calculate_layer_importance(model.to(args.device), example_prompts)
        score_ldx = []
        # Print the importance scores
        for idx, score in enumerate(importance_scores):
            score_ldx.append(score)
            print(f"Layer {idx + 1} Importance: {score:.4f}")

        print(score_ldx)
        torch.save(score_ldx, '/public/ly/SBF/pruning_method/BI/bookcorpus_{}.pth'.format(args.base_model))


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
    ## CUDA_VISIBLE_DEVICES=1,2 TRANSFORMERS_OFFLINE=1 python pruning_method.py --base_model llama3-8b  --pruning_method magnitude_l1  --norm_power 1
    ## CUDA_VISIBLE_DEVICES=1,2 TRANSFORMERS_OFFLINE=1 python pruning_method.py --base_model llama3-8b  --pruning_method magnitude_l2  --norm_power 2
    ## CUDA_VISIBLE_DEVICES=1,2 TRANSFORMERS_OFFLINE=1 python pruning_method.py --base_model llama3-8b  --pruning_method ppl
    main(args)
