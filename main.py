from indexer import create_index
# main.py
from model_loader import load_model
from query import query_model
from index_builder import create_index
import torch


def main():
    tokenizer, model = load_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    index = create_index()  # Create the index from the public corpus

    print("\n🔹 Type 'exit' to stop.")
    while True:
        user_prompt = input("\n💬 Enter your query: ")
        if user_prompt.lower() == "exit":
            print("👋 Exiting. Have a great day!")
            break

        response = query_model(user_prompt, model, tokenizer, device, index)
        print("\n🤖 AI Response:", response)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
