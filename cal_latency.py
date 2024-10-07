import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pruning_method import PPLMetric

parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
parser.add_argument('--model_path', type=str, help='prune model name')
parser.add_argument('--device', type=str, default="cuda", help='device')
args = parser.parse_args()
torch_version = int(torch.__version__.split('.')[1])
args.torch_version = torch_version


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    print('using ddp...')
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

tokenizer = AutoTokenizer.from_pretrained(args.model_path,
    use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(args.model_path,
    trust_remote_code=True, device_map=device_map
)

print(model)

tokenizer.pad_token_id = (0)
tokenizer.padding_side = "left"

start = time.time()
ppl = PPLMetric(model.to(args.device), tokenizer, ['wikitext2'], 128, device=args.device, batch_size=10)

end = time.time()
print(end-start)