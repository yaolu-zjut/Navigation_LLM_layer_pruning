# Navigation-LLM-layer-pruning
## Introduction
Although large language models (LLMs) have achieved remarkable success across various domains, their considerable scale necessitates substantial computational resources, posing significant challenges for deployment in resource-constrained environments. Layer pruning, as a simple yet effective compression method, removes layers of a model directly, reducing computational overhead. However, what are the best practices for layer pruning in LLMs? Are sophisticated layer selection metrics truly effective? Does the LoRA (Low-Rank Approximation) family, widely regarded as a leading method for pruned model fine-tuning, truly meet expectations when applied to post-pruning fine-tuning? To answer these questions, we dedicate thousands of GPU hours to benchmarking layer pruning in LLMs and gaining insights across multiple dimensions. Our results demonstrate that a simple approach, i.e., pruning the final 25\% of layers followed by fine-tuning the \texttt{lm\_head} and the remaining last three layer, yields remarkably strong performance. Following this guide, we prune Llama-3.1-8B-It and obtain a model that outperforms many popular LLMs of similar size, such as ChatGLM2-6B, Vicuna-7B-v1.5, Qwen1.5-7B and Baichuan2-7B. We release the optimal model weights on Huggingface, and the code is available on GitHub.

### Supported LLMs:
- [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [Qwen1.5-7B](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwim-qfT1IaJAxUNr1YBHU-wF8UQFnoECB4QAQ&url=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen1.5-7B&usg=AOvVaw2E2lUSV7wML81PPxhzIfqJ&opi=89978449)
- [Gemma2-2B-It](https://huggingface.co/google/gemma-2-2b-it)
- [Llama-3.1-8B-It](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Our Pruned Models
- [Llama-3.1-6.3B-It-Alpaca](https://huggingface.co/anonymousICLR/Llama-3.1-6.3B-It-Alpaca) 
- [Llama-3.1-6.3B-It-Dolly](https://huggingface.co/anonymousICLR/Llama-3.1-6.3B-It-Dolly/)


## Step-by-step Instructions
**1. Download Hellaswag from Huggingface:**
```python
python hf_download.py --dataset  Rowan/hellaswag  --save_dir saved_path
```

**2. Download Vicuna-7b-v1.5 from Huggingface:**
```python
python hf_download.py --model  lmsys/vicuna-7b-v1.5  --save_dir saved_path
```

**3. Llama-3.1-8B-Instruct Pruning with 8 layers pruned using reverse-order:**

```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python prune_llm.py --base_model Llama-3.1-8B-Instruct --save_model  --pr_method tail --remove_layer 8
```

**4. Llama-3.1-8B-Instruct Finetuning with LoRA:**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python finetune_pruned.py --base_model Llama-3.1-8B-Instruct --save_model --pr_method  tail  --remove_layer 8 --prune_model_path your_path
```

**5. Llama-3.1-8B-Instruct Finetuning with Partial-Layer Finetuning:**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python partial_fine-tuning.py --base_model Llama-3.1-8B-Instruct --save_model  --prune_model_path your_path  --partial_layer_name last3
```
**6. Evaluating the Performance of the Pruned Llama-3.1-8B-Instruct (with LoRA) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):**
```python
#CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 lm_eval --model hf  --model_args pretrained=model_path,trust_remote_code=True,peft=lora_path,parallelize=True --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge  --device cuda:0  --batch_size auto  --num_fewshot 0
```

**7. Evaluating the Performance of the Pruned Llama-3.1-8B-Instruct (with Partial-Layer Finetuning) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 lm_eval --model hf  --model_args pretrained=model_path,trust_remote_code=True,parallelize=True --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge  --device cuda:0  --batch_size auto  --num_fewshot 0
```

**8. Testing MACs, Params and Memory:**
```python
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python test_speedup.py --base_model_path model_path
```

### Zero-shot Evaluation

## Acknowledgement
- The evaluation of the LLM: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Code Framework: https://github.com/horseee/LLM-Pruner 
