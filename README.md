# Navigation-LLM-layer-pruning

## Step-by-step Instructions
**1. Download Hellaswag from Huggingface:**
```python
python hf_download.py --dataset  Rowan/hellaswag  --save_dir saved_path
```

**1. Download Vicuna-7b-v1.5 from Huggingface:**
```python
python hf_download.py --model  lmsys/vicuna-7b-v1.5  --save_dir saved_path
```

**. Llama-3.1-8B-Instruct Pruning with 8 layers pruned using reverse-order:**

```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python prune_llm.py --base_model Llama-3.1-8B-Instruct --save_model  --pr_method tail --remove_layer 8
```


**2. Llama-3.1-8B-Instruct Finetuning with LoRA:**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python finetune_pruned.py --base_model Llama-3.1-8B-Instruct --save_model --pr_method  tail  --remove_layer 8 --prune_model_path your_path
```

**3. Llama-3.1-8B-Instruct Finetuning with Partial-Layer Finetuning:**
```python
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 python partial_fine-tuning.py --base_model Llama-3.1-8B-Instruct --save_model  --prune_model_path your_path  --partial_layer_name last3
```
