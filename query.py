import torch
from model_loader import load_model


def query_model(prompt):
    tokenizer, model = load_model()

    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available

    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to MPS/CPU
    outputs = model.generate(**inputs,
                             max_new_tokens=50,
                             do_sample=True,  # Enables faster token sampling
                             temperature=0.7  # Adjusts randomness (lower = more deterministic)
                             )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
