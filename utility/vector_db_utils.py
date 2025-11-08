import os
from typing import Tuple, Optional
from urllib.parse import urlparse

import requests
from datasets import load_dataset

from utility.logger import logger


def parse_source_path(source_path: str) -> Tuple[str, str]:
    """
    Parses the source path to determine its type and extract relevant information.

    Returns:
        Tuple[str, str]: (source_type, parsed_name_or_path)
                         where source_type is one of: 'url', 'hf', 'local'
    """
    logger.info(f"Parsing source path: {source_path}")

    if source_path.startswith("http://") or source_path.startswith("https://"):
        corpus_name = source_path.rstrip("/").split("/")[-1]
        logger.info(f"Source type identified as URL with corpus name: {corpus_name}")
        return "url", corpus_name

    elif ":" in source_path and not os.path.exists(source_path):
        # Hugging Face dataset format
        dataset_config = source_path.replace(":", "_")
        logger.info(f"Source type identified as HF with config: {dataset_config}")
        return "hf", dataset_config

    else:
        # Local file system path - always return absolute path and folder name
        abs_path = os.path.abspath(source_path)
        folder_name = os.path.basename(os.path.normpath(abs_path))
        logger.info(f"Source type identified as local path: {abs_path}")
        return "local", folder_name


def validate_url(url: str) -> bool:
    """
    Validates the URL to ensure it is safe and matches expected patterns.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.

    Raises:
        ValueError: If the URL is invalid or unsafe.
    """
    logger.info(f"Validating URL: {url}")
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        logger.error("URL must use HTTPS.")
        raise ValueError("URL must use HTTPS.")
    allowed_domains = ["example.com", "trusted-source.com"]
    if parsed_url.netloc not in allowed_domains:
        logger.error(f"Domain '{parsed_url.netloc}' is not allowed.")
        raise ValueError(f"Domain '{parsed_url.netloc}' is not allowed.")
    return True


def download_and_save_from_url(url: str, target_dir: str) -> None:
    """
    Downloads text data from a URL and saves it locally using streaming.

    Args:
        url (str): URL to download the data from.
        target_dir (str): Directory to save the downloaded data.

    Returns:
        None
    """
    validate_url(url)
    logger.info(f"Downloading data from URL: {url}")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, "corpus.txt")

    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            logger.error(f"Failed to download from {url}, status code: {response.status_code}")
            raise Exception(f"Failed to download from {url}, status code: {response.status_code}")
        with open(file_path, "w", encoding="utf-8") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk.decode("utf-8"))
    logger.info(f"Downloaded and saved corpus to {file_path}")


def download_and_save_from_hf(dataset_name: str, config: Optional[str], target_dir: str,
                              max_docs: Optional[int] = None) -> None:
    """
    Downloads a dataset from Hugging Face and saves it locally as text files in batches.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        config (Optional[str]): Configuration for the dataset (can be None).
        target_dir (str): Directory to save the downloaded documents.
        max_docs (int): Maximum number of documents to download.

    Returns:
        None
    """
    if config:
        logger.info(f"Downloading dataset '{dataset_name}' with config '{config}' from Hugging Face...")
        dataset_identifier = f"{dataset_name}:{config}"
    else:
        logger.info(f"Downloading dataset '{dataset_name}' from Hugging Face...")
        dataset_identifier = dataset_name

    try:
        if config:
            dataset = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)

        os.makedirs(target_dir, exist_ok=True)

        batch_size = 100
        batch = []
        doc_count = 0

        # Heuristic to find a text column if 'text' isn't present
        text_column = None
        if 'text' in dataset.column_names:
            text_column = 'text'
        else:
            # Try to find a reasonable text column
            for col_name in dataset.column_names:
                if 'text' in col_name.lower() or 'document' in col_name.lower() or 'content' in col_name.lower():
                    # Check if the first item in this column is a string
                    if len(dataset) > 0 and isinstance(dataset[0][col_name], str):
                        text_column = col_name
                        logger.warning(f"Using column '{text_column}' as text content for HF dataset.")
                        break

            if text_column is None:
                raise ValueError(f"No obvious text column found in dataset '{dataset_identifier}'. "
                                 f"Available columns: {dataset.column_names}. Please specify manually if possible.")

        for i, item in enumerate(dataset):
            if max_docs is not None and doc_count >= max_docs:
                break

            text_content = item.get(text_column, "").strip()

            if text_content:  # Only process if text content is not empty
                batch.append((doc_count, text_content))
                doc_count += 1

            if len(batch) >= batch_size or (max_docs is not None and doc_count >= max_docs):
                for doc_id, doc_text in batch:
                    with open(os.path.join(target_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
                        f.write(doc_text)
                batch.clear()

        # Save any remaining documents in the last batch
        if batch:
            for doc_id, doc_text in batch:
                with open(os.path.join(target_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(doc_text)
            batch.clear()

        logger.info(f"Saved {doc_count} documents to {target_dir}")

    except Exception as e:
        logger.error(f"Failed to download or process HF dataset '{dataset_identifier}': {e}")
        raise


import concurrent.futures
from tqdm import tqdm
import time


def load_local_dataset(local_dir: str):
    """
    Load a local dataset directory either as a Hugging Face dataset or fallback to SimpleDirectoryReader.
    Now supports parallel processing and real-time progress display without caching.
    """
    import json
    import datasets
    from llama_index.core import Document
    from llama_index.core import SimpleDirectoryReader

    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Attempting to load local dataset from directory: {local_dir}")
    dataset_dict_path = os.path.join(PROJECT_PATH, "data", local_dir, "dataset_dict.json")
    documents = []

    def _process_example(example):
        for col in ['text', 'abstract', 'context', 'content']:
            if col in example and example[col]:
                return Document(text=example[col])
        return None

    try:
        if os.path.exists(dataset_dict_path):
            print(f"Found dataset_dict.json in {local_dir}, loading as Hugging Face dataset.")
            logger.info(f"Found dataset_dict.json in {local_dir}, loading as Hugging Face dataset.")
            with open(dataset_dict_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)

            splits = dataset_info.get("splits", ["train"])
            logger.info(f"Loading splits: {splits}")

            for split in splits:
                split_dir = os.path.join(PROJECT_PATH, "data", local_dir, split)
                if not os.path.exists(split_dir):
                    logger.warning(f"Split directory '{split_dir}' does not exist, skipping.")
                    continue

                ds = datasets.load_from_disk(split_dir)
                total_examples = len(ds)
                logger.info(f"Loaded split '{split}' with {total_examples} examples.")

                split_docs = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    for doc in tqdm(executor.map(_process_example, ds),
                                    total=total_examples,
                                    desc=f"Processing split '{split}'"):
                        if doc:
                            split_docs.append(doc)
                documents.extend(split_docs)

            logger.info(f"✅ Loaded total {len(documents)} documents from local Hugging Face dataset.")
            print(f"✅ Finished loading {len(documents)} documents from {local_dir}")
            return documents

        else:
            print(f"No dataset_dict.json found in {local_dir}, falling back to SimpleDirectoryReader.")
            reader = SimpleDirectoryReader(local_dir)
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents using SimpleDirectoryReader.")
            return documents

    except Exception as e:
        logger.error(f"Error loading local dataset: {e}")
        raise
