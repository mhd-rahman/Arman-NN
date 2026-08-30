"""Test loading ArmanNN from HuggingFace Hub."""

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model from HuggingFace Hub...")
model = AutoModelForCausalLM.from_pretrained(
    "mhd-rahman/ArmanNN-Base", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("mhd-rahman/ArmanNN-Base")

print(f"Model loaded! Device: {next(model.parameters()).device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test generation
prompt = "The capital of France is"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8, top_p=0.9)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated: {generated}")
