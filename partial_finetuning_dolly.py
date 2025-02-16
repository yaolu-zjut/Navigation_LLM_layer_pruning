'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import os
import random
import argparse
import numpy as np
import torch
import transformers
from functools import partial
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from utils.utils import prepare_model_for_int8_training
from typing import Any, Dict, List, Tuple, Union
from trl import SFTTrainer
from utils.consts import (
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    RESPONSE_KEY_NL,
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"


### not suitable for gemma2
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def main(args):
    # Load Pruned Model
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
                                              use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
                                                 trust_remote_code=True, device_map=device_map)

    print(model)
    for param in model.parameters():
        param.requires_grad = False

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

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    # Prepare For LoRA
    print('model is ready...')


    def load_training_dataset(path_or_dataset):
        dataset = load_dataset(path_or_dataset)["train"]
        print("Found %d rows", dataset.num_rows)

        def _add_text(rec):
            instruction = rec["instruction"]
            response = rec["response"]
            context = rec.get("context")

            if not instruction:
                raise ValueError(f"Expected an instruction in: {rec}")

            if not response:
                raise ValueError(f"Expected a response in: {rec}")

            # For some instructions there is an input that goes along with the instruction, providing context for the
            # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
            # some piece of information from it.  The response is that information to extract.  In other cases there is
            # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
            # born.
            if context:
                rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
            else:
                rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
            return rec

        dataset = dataset.map(_add_text)

        return dataset

    def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )

    def preprocess_dataset(tokenizer, max_length, seed, training_dataset):
        """Loads the training dataset and tokenizes it so it is ready for training.

        Args:
            tokenizer (AutoTokenizer): Tokenizer tied to the model.
            max_length (int): Maximum number of tokens to emit from tokenizer.

        Returns:
            Dataset: HuggingFace dataset
        """

        dataset = load_training_dataset(training_dataset)

        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["instruction", "context", "response", "text", "category"],
        )

        dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
        dataset = dataset.shuffle(seed=seed)

        return dataset

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=1024, seed=42,
                                           training_dataset='/public/home/dataset/databricks-dolly-15k/')

    split_dataset = processed_dataset.train_test_split(test_size=200, seed=42)

    print("Train data size: %d", split_dataset["train"].num_rows)
    print("Test data size: %d", split_dataset["test"].num_rows)

    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        dataset_text_field="text",
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            logging_first_step=True,
            fp16=True,
            bf16=False,
            eval_steps=100,
            save_steps=200,
            save_total_limit=20,
            max_grad_norm=1.0,
            output_dir=args.output_dir,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="eval_loss",
        ),
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        ),
    )
    model.config.use_cache = False

    trainer.train()
    # model = model.merge_and_unload()

    if args.save_model:
        output_lora_dir = '/public/model/dolly/partial_finetuned_dolly_{}_{}{}/'.format(
            args.base_model, args.pr_method, args.remove_layer)
        if not os.path.exists(output_lora_dir):
            os.mkdir(output_lora_dir)
        model.save_pretrained(output_lora_dir)
        tokenizer.save_pretrained(output_lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="llama3-8b", help='base model name')
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--remove_layer', type=int, default=16, help='batch size')
    parser.add_argument('--output_dir', type=str,
                        default="",
                        help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')  # 2
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=1024, help='cutoff length')
    parser.add_argument('--partial_layer_name', type=str, default="last3", help='base model name')
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
