from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load LLaMA-3.1 model
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with actual model name if different
#model_name = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Determine all layers to apply LoRA
target_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):  # Apply LoRA to all linear layers
        print(name, module.in_features, module.out_features)
        target_modules.append(name)

# Function to get the maximum rank for each layer
"""def get_max_rank(model, target_modules):
    max_ranks = {}
    for name, module in model.named_modules():
        print(" NAME ", name)
        if name in target_modules:
            print("Applying Lora")
            max_ranks[name] = min(module.in_features, module.out_features)  # Max rank
    return max_ranks"""

# Get maximum rank for each layer
#max_ranks = get_max_rank(model, target_modules)
# 24784128000
# Define LoRA configuration for maximum ranks
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type
    inference_mode=False,
    r=256,  # Set rank to the maximum observed
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout
    target_modules=target_modules  # Apply to all identified linear layers
)

# Add LoRA adapters to the model
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# Count parameters introduced by LoRA
"""def count_lora_params(model):
    total_params = 0
    normal_params = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            total_params += param.numel()
        else:
            normal_params += param.numel()
    return total_params, normal_params

# Calculate added parameters
added_params, previous_params = count_lora_params(lora_model)
print(f"LoRA added {added_params} parameters. Originally {previous_params}")"""
