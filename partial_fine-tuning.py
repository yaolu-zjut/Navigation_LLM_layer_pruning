'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import gc
import os
import random
import argparse
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompter import Prompter, ZeroPrompter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["WANDB_DISABLED"]="true"

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## For clearing GPU memory
def clear_gpu_memory(debug=False):
    """
    Clears GPU memory on all devices and optionally provides debugging information about memory usage.

    Args:
    debug (bool): If True, print memory stats before and after cleanup.
    """
    if debug:
        print("Before cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")

    # Collect garbage to potentially free up memory references
    gc.collect()

    # Clear PyTorch's CUDA memory cache
    torch.cuda.empty_cache()

    if debug:
        print("After cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")


def main(args):
    # Load Pruned Model
    set_random_seed(args.seed)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('using ddp...')
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer = AutoTokenizer.from_pretrained(args.prune_model_path,
        use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
        trust_remote_code=True, device_map=device_map
    )

    print(model)
    for param in model.parameters():
        param.requires_grad = False

    ### for lm_head and norm, not suitable for gemma
    for param in model.model.norm.parameters():
        param.requires_grad = True

    for param in model.lm_head.parameters():
        param.requires_grad = True

    if args.partial_layer_name == 'last1':
        for param in model.model.layers[-1].parameters():
            param.requires_grad = True
    elif args.partial_layer_name == 'last2':
        for param in model.model.layers[-1].parameters():
            param.requires_grad = True
        for param in model.model.layers[-2].parameters():
            param.requires_grad = True
    elif args.partial_layer_name == 'last3':
        for param in model.model.layers[-1].parameters():
            param.requires_grad = True
        for param in model.model.layers[-2].parameters():
            param.requires_grad = True
        for param in model.model.layers[-3].parameters():
            param.requires_grad = True
    elif args.partial_layer_name == 'norm_lmhead':
        print('just finetune norm and lm_head')


    for name, param in model.named_parameters():
        print(f"Layer: {name}, requires_grad: {param.requires_grad}")

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            # print('using alpaca...')
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

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

    clear_gpu_memory()

    # Load Train Dataset
    data = load_dataset(args.data_path)
    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = {
            args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from evaluate.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128  # too small, ori 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,  # 100 ori
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,  # not torch.cuda.is_bf16_supported()
            bf16=False,  # torch.cuda.is_bf16_supported()
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            max_grad_norm=1.0,
            report_to="none",
            load_best_model_at_end=True,
            # lr_scheduler_type="linear",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()

    if args.save_model:
        output_lora_dir = 'LLM_pretrained/pruned_model/partial_tuing_taylor8/partial_tuing_{}_alpaca_{}/'.format(args.base_model, args.partial_layer_name)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        model.save_pretrained(output_lora_dir)
        tokenizer.save_pretrained(output_lora_dir)

    clear_gpu_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="/public/MountData/dataset/alpaca-cleaned/", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str,
                        default="LLM_pretrained/pruned_model/lora-alpaca-llama/",
                        help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')  # 2 or 5
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca",
                        help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False,
                        help="Whether to use the instruction template or not.")
    parser.add_argument('--partial_layer_name', type=str, default="last1", help='base model name')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true",
                        help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true",
                        help="faster, but produces an odd training loss curve")

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    ## CUDA_VISIBLE_DEVICES=2 TRANSFORMERS_OFFLINE=1 python partial_fine-tuning.py --base_model Qwen1.5-7B --save_model  --prune_model_path ~  --partial_layer_name ~
    main(args)
