import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
load_in_4bit = True

print("⌛ Loading Qwen 2.5 7B in 4-bit mode...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# For DAPT, we use a higher rank (r=32) so the model can learn complex raw domain terminology
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Higher rank for DAPT
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
print("✅ Base Model ready for Continued Pre-Training!")
