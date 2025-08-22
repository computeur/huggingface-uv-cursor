import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "sshleifer/tiny-gpt2"  # tiny demo model

print(f"Using device: {device}")
print(f"Loading model: {model_id}")

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

inp = tok("Hello from Cursor + HF:", return_tensors="pt").to(device)
out = model.generate(**inp, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
