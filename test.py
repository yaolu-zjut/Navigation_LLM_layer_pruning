import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model

from utils.eval import eval_zero_shot

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
a =torch.load('/public/ly/SBF/pruning_method/BI/bookcorpus_5data_Vicuna_7B.pth')
print(a)
print(len(a))
# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )
#
# base_model = AutoModelForCausalLM.from_pretrained('/public/MountData/yaolu/LLM_pretrained/pruned_model/pruned_Vicuna_7B_tail/', trust_remote_code=True, quantization_config=nf4_config, device_map="auto")
#
# lora_config = LoraConfig.from_pretrained('/public/MountData/yaolu/LLM_pretrained/pruned_model/finetuned_Qlora_alpaca_Vicuna_7B_tail/')
# model = get_peft_model(base_model, lora_config)
# # tasks = ['piqa', 'arc_challenge', 'arc_easy']
# print(type(base_model))
# print(type(model))
# tasks = ['arc_easy']
# results = eval_zero_shot(model_name='finetuned_Qlora_alpaca_Vicuna_7B_tail', model=base_model, task_list=tasks, parallelize=True)
# results = results['results']
#
# for task in tasks:
#     print(f"{task}: {results[task]}")


# import torch
#
# # #######  相似性分块代码  ##########
# import numpy as np
#
#
# def calculate_cost(dissimilarity, start, end):
#     return np.sum(dissimilarity[start:end, start:end])
#
#
# def divide_matrix(matrix, num_blocks=5):
#     n = matrix.shape[0]
#
#     dp = np.full((n + 1, num_blocks + 1), np.inf)
#     dp[0][0] = 0
#
#     for j in range(1, num_blocks + 1):
#         for i in range(1, n + 1):
#             for k in range(i):
#                 cost = calculate_cost(matrix, k, i)
#                 dp[i][j] = min(dp[i][j], dp[k][j - 1] + cost)
#
#     # Traceback to find the actual blocks
#     blocks = []
#     i = n
#     for j in range(num_blocks, 0, -1):
#         for k in range(i):
#             cost = calculate_cost(matrix, k, i)
#             if dp[i][j] == dp[k][j - 1] + cost:
#                 blocks.append(list(range(k, i)))
#                 i = k
#                 break
#
#     return blocks[::-1]
#
#
# # Example usage
# for i in range(20):
#     similarity_matrix = torch.load('/public/ly/SBF/img/sim_matrix_llama3-8b_id{}_batch8.pth'.format(i))
#     blocks = divide_matrix(1-similarity_matrix.numpy(), 6)
#     for i, block in enumerate(blocks):
#         print(f"Block {i + 1}: Rows {block}")
#
#     print('---'*50)

raw_dataset = load_dataset('allenai/c4', data_files="en/c4-train.00000-of-01024.json.gz", split='train[:1%]')




