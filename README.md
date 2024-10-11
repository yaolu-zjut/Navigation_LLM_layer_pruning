# Navigation-LLM-layer-pruning
## 
### Supported LLMs:
- [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [Qwen1.5-7B](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwim-qfT1IaJAxUNr1YBHU-wF8UQFnoECB4QAQ&url=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen1.5-7B&usg=AOvVaw2E2lUSV7wML81PPxhzIfqJ&opi=89978449)
- [Gemma2-2B-It](https://huggingface.co/google/gemma-2-2b-it)
- [Llama-3.1-8B-It](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)


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
