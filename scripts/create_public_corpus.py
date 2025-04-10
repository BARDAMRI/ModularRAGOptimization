import os
from datasets import load_dataset

# Load a small public corpus from the "wikitext" dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Directory where we'll store the text files
data_dir = "../data/public_corpus"
os.makedirs(data_dir, exist_ok=True)

# Save each non-empty document to a separate text file
for i, example in enumerate(dataset):
    text = example["text"]
    if text.strip():  # only write non-empty documents
        with open(os.path.join(data_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    print(f">Saved {i + 1} documents to '{data_dir}'")
