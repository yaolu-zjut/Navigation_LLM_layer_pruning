from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

from pruning_method import PPLMetric

device_map ='auto'


tokenizer_front = AutoTokenizer.from_pretrained(
    '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
    use_fast=False, trust_remote_code=True
)
model_front = AutoModelForCausalLM.from_pretrained(
    '/public/MountData/yaolu/LLM_pretrained/Vicuna_7B_V1.5/models--lmsys--vicuna-7b-v1.5/',
    trust_remote_code=True, device_map=device_map, use_cache=False
)

tokenizer_later = AutoTokenizer.from_pretrained(
    '/public/MountData/yaolu/LLM_pretrained/gemma-2-2b-it/',
    use_fast=False, trust_remote_code=True
)
model_later = AutoModelForCausalLM.from_pretrained(
    '/public/MountData/yaolu/LLM_pretrained/gemma-2-2b-it/',
    trust_remote_code=True, device_map=device_map, use_cache=False
)
print(model_front)
embed_tokens = model_front.model.embed_tokens

# 获取llama3的前8层
model_front_layers = model_front.model.layers[:8]
print(model_front_layers)

# 获取vicana的后16层
model_later_layers = model_later.model.layers[-16:]
print(model_later_layers)

class CombinedModel(nn.Module):
    def __init__(self, model_embed_tokens, model_front_layers, model_later_layers, stitching=False):
        super(CombinedModel, self).__init__()
        self.embed_tokens = model_embed_tokens
        self.front_layers = nn.ModuleList(model_front_layers)
        self.later_layers = nn.ModuleList(model_later_layers)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.front_layers:
            x = layer(x)
        print(x.shape)
        for layer in self.later_layers:
            x = layer(x)
        return x

combined_model = CombinedModel(embed_tokens, model_front_layers, model_later_layers)

ppl = PPLMetric(combined_model.cuda(), tokenizer_front, ['wikitext2'], 128, device="cuda")
print("PPL before pruning: {}".format(ppl))