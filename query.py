import torch
from model_loader import load_model


def query_model(prompt):
    tokenizer, model = load_model()

    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS if available

    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to MPS/CPU
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    user_prompt = input("Enter your query: ")
    print(query_model(user_prompt))
