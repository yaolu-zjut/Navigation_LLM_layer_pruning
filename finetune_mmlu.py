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
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from utils.utils import prepare_model_for_int8_training

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["WANDB_DISABLED"]="true"

device = "cuda" if torch.cuda.is_available() else "cpu"

### not suitable for gemma2
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Load Pruned Model
    set_random_seed(args.seed)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

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
        trust_remote_code=True, device_map=device_map, num_labels=4
    )  # 4 chioce

    print(model)

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    def preprocess_function(examples):
        inputs = []
        labels = []

        # Loop over each example in the dataset
        for question, choices, answer in zip(examples['question'], examples['choices'], examples['answer']):
            # Concatenate the question and the choices into a single input
            input_text = question + " " + " ".join([f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])
            inputs.append(input_text)

            # Check if the answer is already an integer (e.g., 0, 1, 2, 3 for choices A, B, C, D)
            if isinstance(answer, int):
                answer_index = answer
            # If the answer is a string (like 'A', 'B', 'C', 'D'), convert it to an index
            elif isinstance(answer, str) and len(answer) == 1:
                answer_index = ord(answer) - ord('A')
            else:
                raise ValueError(f"Unexpected format for answer: {answer}")

            labels.append(answer_index)

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

        # Use the labels as the index of the correct choice
        model_inputs["labels"] = labels
        return model_inputs

    # Apply the preprocess function to the dataset
    dataset = load_dataset("/public/ly/SBF/evaluate/mmlu_with_train.py", "all")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    print('model is ready...')
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    from transformers import Trainer, DataCollatorWithPadding

    class MultipleChoiceTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Inspect logits shape
            print("Logits shape before reshaping:", logits.shape)

            # Get batch size and number of choices from inputs
            batch_size = inputs["input_ids"].size(0)  # This is 4 in your case
            num_choices = inputs["input_ids"].size(1)  # Number of choices (this could be 4 or more)

            # Reshape logits (might need to handle additional dimensions)
            logits = logits.view(batch_size, num_choices, -1)

            # Optionally, reduce to `[batch_size, num_choices]`
            logits = logits.mean(dim=-1)  # Averaging over hidden dimensions (this is one strategy)

            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Use the customized trainer in place of the original Trainer
    trainer = MultipleChoiceTrainer(
        model=model,
        train_dataset=tokenized_dataset["auxiliary_train"],
        eval_dataset=tokenized_dataset["validation"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            bf16=False,
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
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="eval_loss",
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    model.config.use_cache = False

    trainer.train()
    # model = model.merge_and_unload()

    if args.save_model:
        output_lora_dir = '/public/MountData/yaolu/LLM_pretrained/pruned_model/oneshot/taylor/finetuned_lora_alpaca_{}_{}{}/'.format(args.base_model, args.pr_method, args.remove_layer)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        model.save_pretrained(output_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--remove_layer', type=int, default=16, help='batch size')
    parser.add_argument('--output_dir', type=str,
                        default="/public/MountData/yaolu/LLM_pretrained/pruned_model/finetuned_lora_alpaca-llama/",
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
    ## CUDA_VISIBLE_DEVICES=0,2 TRANSFORMERS_OFFLINE=1 python finetune_pruned.py --base_model Qwen1.5-7B --save_model --pr_method  magnitude_l1  --remove_layer 8 --prune_model_path ~
    main(args)
