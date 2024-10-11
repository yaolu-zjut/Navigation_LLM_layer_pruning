# Navigation-LLM-layer-pruning

## Step-by-step Instructions
**1. Llama-3.1-8B-Instruct Pruning with 8 layers pruned using reverse-order:**

```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python prune_llm.py --base_model Llama-3.1-8B-Instruct --save_model  --pr_method tail --remove_layer 8
```


**2. Llama-3.1-8B-Instruct Finetuning with LoRA:**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python finetune_pruned.py --base_model Llama-3.1-8B-Instruct --save_model --pr_method  tail  --remove_layer 8 --prune_model_path your_path
```

