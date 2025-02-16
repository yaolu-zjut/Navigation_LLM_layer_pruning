import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
import transformers
from trl import SFTTrainer
from utils.prompter import Prompter, ZeroPrompter

os.environ["WANDB_DISABLED"]="true"
### qlora

def main(args):
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    def find_all_linear_names(model):
      cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
      lora_module_names = set()
      for name, module in model.named_modules():
        if isinstance(module, cls):
          names = name.split('.')
          lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
          lora_module_names.remove('lm_head')
      return list(lora_module_names)


    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.prune_model_path,
        use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
        trust_remote_code=True, quantization_config=bnb_config, device_map=device_map
    )

    print(model)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()


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

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            logging_first_step=True,
            fp16=True,  # not torch.cuda.is_bf16_supported()
            bf16=False,  # torch.cuda.is_bf16_supported()
            eval_steps=100,
            save_steps=200,
            save_total_limit=20,
            max_grad_norm=1.0,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if args.save_model:
        output_lora_dir = 'LLM_pretrained/pruned_model/finetuned_qlora_alpaca_{}_{}{}/'.format(args.base_model, args.pr_method, args.remove_layer)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        model.save_pretrained(output_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="/public/MountData/dataset/alpaca-cleaned/", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--remove_layer', type=int, default=16, help='batch size')
    parser.add_argument('--output_dir', type=str,
                        default="LLM_pretrained/pruned_model/finetuned_lora_alpaca-llama/",
                        help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')  # 2
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca",
                        help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False,
                        help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

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
    parser.add_argument('--pr_method', type=str, default="ppl", help='device')

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    ## CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python finetune_pruned_qlora.py --base_model  Gemma2-2b  --save_model --pr_method random --prune_model_path  
    main(args)
