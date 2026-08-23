import math

FastLanguageModel.for_inference(model)

test_domain_text = "The Peng-Robinson Equation of State models gas non-idealities. Arrhenius rate constant k(T) = A * exp(-Ea / (R*T))."

inputs = tokenizer(test_domain_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(inputs.input_ids, labels=inputs.input_ids)
    loss = outputs.loss.item()
    perplexity = math.exp(loss)

print("="*60)
print("📊 DAPT EVALUATION RESULTS:")
print("="*60)
print(f"📉 Cross-Entropy Loss : {loss:.4f}")
print(f"📉 Model Perplexity    : {perplexity:.2f}")
print("\n(Note: Lower Perplexity indicates higher domain fluency and lower surprise when reading domain terms!)")
