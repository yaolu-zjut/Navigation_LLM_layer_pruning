import argparse
import torch
import numpy as np
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.activations import silu
from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import pool_flops_counter_hook
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])


def LlamaAttention_counter_hook(module, input, output):
    # (1) Ignore past-key values
    # (2) Assume there is no attention mask
    # Input will be empty in some pytorch version. use output here since input.shape == output.shape
    flops = 0
    q_len = output[0].shape[1]
    linear_dim = output[0].shape[-1]
    num_heads = module.num_heads
    head_dim = module.head_dim

    rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
    attention_flops = num_heads * (
                q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim)  # QK^T + softmax + AttentionV
    linear_flops = 4 * (q_len * linear_dim * num_heads * head_dim)  # 4 for q, k, v, o.
    flops += rotary_flops + attention_flops + linear_flops
    module.__flops__ += int(flops)


def rmsnorm_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    batch_flops *= 2
    module.__flops__ += int(batch_flops)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, trust_remote_code=True, device_map='auto')

    def input_constructor(x):
        return {'input_ids': torch.ones(x).long().to(device)}

    if device == "cuda":
        model = model.cuda()

        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                     input_constructor=input_constructor,
                                                     print_per_layer_stat=True, verbose=True,
                                                     custom_modules_hooks={
                                                         LlamaAttention: LlamaAttention_counter_hook,
                                                         LlamaRMSNorm: rmsnorm_flops_counter_hook,
                                                         silu: pool_flops_counter_hook,
                                                     }, )
    else:
        model.float()
        macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                 input_constructor=input_constructor,
                                                 print_per_layer_stat=True, verbose=True,
                                                 custom_modules_hooks={
                                                     LlamaAttention: LlamaAttention_counter_hook,
                                                     LlamaRMSNorm: rmsnorm_flops_counter_hook,
                                                     silu: pool_flops_counter_hook,
                                                 }, )

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated() / 1024 / 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')
    parser.add_argument('--base_model_path', type=str, default="", help='base model name')

    args = parser.parse_args()
    main(args)
