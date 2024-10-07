import argparse

import torch
import matplotlib.pyplot as plt
from peft import PeftModel
from transformers import AutoModelForCausalLM
import numpy as np


# Function to extract and plot weight distributions
def plot_weight_distribution(model):
    weights = []

    # Traverse all named parameters in the model
    for name, param in model.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())  # Flatten and convert to NumPy array

    # Concatenate all weights into a single array
    all_weights = np.concatenate(weights)

    # Plotting the histogram of the weights
    plt.figure(figsize=(12, 6))
    plt.hist(all_weights, bins=100, range=(-0.2, 0.2), color='blue', alpha=0.7)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('Weight Distribution of Iter gemma taylor12.pdf')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')
    parser.add_argument('--prune_model_path', type=str, help='prune model name')
    parser.add_argument('--lora_path', type=str, help='lora name')
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    model = AutoModelForCausalLM.from_pretrained(args.prune_model_path,
                                                 trust_remote_code=True, device_map='auto'
                                                 )
    lora_model = PeftModel.from_pretrained(
        model,
        args.lora_path,
        torch_dtype=torch.float16,
    )
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    plot_weight_distribution(model)
