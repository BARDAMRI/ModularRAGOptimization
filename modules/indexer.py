from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from config import DATA_PATH, INDEX_SOURCE_URL, HF_MODEL_NAME
import os
import requests
from datasets import load_dataset


def download_and_save_from_hf(dataset_name, config, target_dir, max_docs=1000):
    dataset = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
    os.makedirs(target_dir, exist_ok=True)

    for i, item in enumerate(dataset):
        if i >= max_docs:
            break
        text = item["text"].strip()
        if text:
            with open(os.path.join(target_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
    print(f"Saved {min(max_docs, len(dataset))} documents to {target_dir}")


def download_and_save_from_url(url, target_dir):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download from {url}, status code: {response.status_code}")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, "corpus.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Downloaded and saved corpus to {file_path}")


def parse_source_path(source_path):
    if "://" in source_path and not source_path.startswith("hf://"):
        return "url", os.path.basename(source_path.rstrip("/"))
    if ":" in source_path:
        dataset, config = source_path.split(":", 1)
    else:
        dataset, config = source_path, None
    return "hf", f"{dataset.replace('/', '_')}_{config or 'default'}"


def load_index(source="local", source_path=None):
    embed_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    Settings.embed_model = embed_model

    if source == "url":
        if source_path is None:
            raise ValueError(
                "source_path must be provided for 'url' source. Please insert into config.py a INDEX_SOURCE_URL "
                "variable with valid data")
        source_type, corpus_name = parse_source_path(source_path)
        corpus_dir = os.path.join("data", corpus_name)
        storage_dir = os.path.join("storage", corpus_name)

        if not os.path.exists(corpus_dir):
            print(f"Downloading corpus into {corpus_dir}...")
            if source_type == "hf":
                dataset, config = source_path.split(":", 1)
                download_and_save_from_hf(dataset, config, corpus_dir)
            else:
                download_and_save_from_url(source_path, corpus_dir)

        if os.path.exists(storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            print(f"Loaded existing index for '{corpus_name}' from {storage_dir}.")
            return index

        documents = SimpleDirectoryReader(corpus_dir).load_data()
        index = GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"Indexed and saved new corpus from URL to {storage_dir}.")
        return index

    else:  # "local"
        storage_dir = "storage"
        if os.path.exists(storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            print("Loaded existing local index from 'storage/'.")
            return index

        documents = SimpleDirectoryReader(DATA_PATH).load_data()
        index = GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=storage_dir)
        print("Indexed and saved new local corpus to 'storage/'.")
        return index


if __name__ == "__main__":
    idx = load_index(source="url", source_path=INDEX_SOURCE_URL)
    print(">Index created with", len(idx.docstore.docs), "documents.")
