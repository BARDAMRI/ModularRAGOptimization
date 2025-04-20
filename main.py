# main.py
from modules.model_loader import load_model
from modules.query import query_model
from modules.indexer import load_index
import torch
from config import DEVICE_TYPE_MPS, DEVICE_TYPE_CPU, INDEX_SOURCE_URL


def main():
    tokenizer, model = load_model()
    device = DEVICE_TYPE_MPS if torch.backends.mps.is_available() else DEVICE_TYPE_CPU

    print("\n> Warming up the model")
    # Warm-up the model to load weights into memory
    _ = model.generate(
        **tokenizer("Warmup", return_tensors="pt").to(device),
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id
    )

    print("\n> Loading external Vector DB ")
    index = load_index(source="url", source_path=INDEX_SOURCE_URL)

    print("\n> Ask anything you want!\nType 'exit' to stop.")
    while True:
        user_prompt = input("\n💬 Enter your query: ")
        if user_prompt.lower() == "exit":
            print("> Exiting. Have a great day!")
            break

        response = query_model(user_prompt, model, tokenizer, device, index)
        print("\n> AI Response:", response)


if __name__ == "__main__":
    main()
