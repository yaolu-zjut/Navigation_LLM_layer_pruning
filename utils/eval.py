import os
import torch


@torch.no_grad()
def eval_zero_shot(model_name, model, task_list=['arc_easy'], num_fewshot=0, parallelize=False, peft=None):
    from lm_eval import tasks, evaluator, utils
    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')

    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = utils.load_yaml_config(task)
            task_names.append(config)

    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True,peft={peft}"

    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size='auto',
        max_batch_size=None,
        device='cuda:0',
        use_cache=None,
        limit=None,
        check_integrity=False,
        write_out=False,
        gen_kwargs=None,
        task_manager=task_manager,
    )

    return results