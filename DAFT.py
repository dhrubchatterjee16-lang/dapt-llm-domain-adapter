from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load raw text dataset
dataset = load_dataset("json", data_files="dapt_corpus.jsonl", split="train")

# Define DAPT Causal LM Training Arguments
training_args = SFTConfig(
    output_dir = "dapt_outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_steps = 40, # 40 steps for Colab demo run
    learning_rate = 1e-4, # Lower learning rate for DAPT
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 5,
    optim = "adamw_8bit",
    output_dir = "dapt_outputs",
)

# SFTTrainer in packing mode runs next-token prediction over raw text fields
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Training on raw text field without chat templates
    max_seq_length = max_seq_length,
    packing = True, # Packs multiple text chunks into single 2048 token windows
    args = training_args,
)

print("⚡ Starting Continued Pre-Training (DAPT)...")
trainer.train()
print("🎉 DAPT Training Complete!")

# Save Domain-Adapted Model Checkpoint
model.save_pretrained("dapt_domain_adapted_lora")
tokenizer.save_pretrained("dapt_domain_adapted_lora")
print("💾 Saved Domain-Adapted Model to 'dapt_domain_adapted_lora'!")
