import argparse
import fnmatch

import lm_eval
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from lm_eval import tasks, utils
import os
import torch


@torch.no_grad()
def eval_zero_shot_for_qlora(args, task_list=['arc_easy', 'mmlu', 'cmmlu', 'piqa', 'openbookqa', 'winogrande', 'hellaswag' , 'arc_challenge']):
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
            trust_remote_code=True, quantization_config=bnb_config, device_map='auto'
        )
    lora_model = PeftModel.from_pretrained(
            model, args.lora_path,
            # torch_dtype=torch.float16,
        )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model)

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager(include_path='/public/MountData/yaolu/lm-evaluation-harness/lm_eval/tasks')

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        batch_size=args.batch_size,  # 'auto'
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None
    )

    print(results['results'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM')

    # Model Type&Path
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lora_path', type=str, help='lora name')

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    eval_zero_shot_for_qlora(args)
    # CUDA_VISIBLE_DEVICES=3 TRANSFORMERS_OFFLINE=1 python test_lm.py --prune_model_path /public/MountData/yaolu/LLM_pretrained/pruned_model/pruned_Vicuna_7B_tail/  --lora_path /public/MountData/yaolu/LLM_pretrained/pruned_model/finetuned_Qlora_alpaca_Vicuna_7B_tail/
    # CUDA_VISIBLE_DEVICES=3 TRANSFORMERS_OFFLINE=1 python test_lm.py --prune_model_path /public/MountData/yaolu/LLM_pretrained/pruned_model/pruned_Vicuna_7B_tail/  --lora_path /public/MountData/yaolu/LLM_pretrained/pruned_model/finetuned_Qlora_alpaca_Vicuna_7B_tail/