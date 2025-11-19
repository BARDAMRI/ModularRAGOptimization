import json
import os
from typing import Tuple, Optional, List
from urllib.parse import urlparse

# Make sure all imports are present
import datasets
import pandas as pd
import requests
from datasets import load_dataset
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader

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


def _load_documents_from_csv(file_path: str,
                             text_column_name: Optional[str] = None,
                             metadata_column_names: Optional[List[str]] = None) -> List[Document]:
    """
    Loads LlamaIndex Documents from a single CSV file.
    Uses 'text_column_name' for text and 'metadata_column_names' for metadata.
    """
    documents = []
    if metadata_column_names is None:
        metadata_column_names = []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {file_path}: {e}")
        return []

    text_col = text_column_name

    # 1. Find text column
    if text_col and text_col in df.columns:
        logger.info(f"Using specified text column: '{text_col}'")
    else:
        if text_col:
            logger.warning(f"Specified text column '{text_col}' not found. Falling back to heuristic.")
        text_col = next(
            (c for c in df.columns if 'text' in c.lower() or 'abstract' in c.lower() or 'content' in c.lower()),
            None)
        if text_col:
            logger.info(f"Using heuristic text column: '{text_col}'")

    if not text_col:
        logger.error(f"No suitable text column found in {file_path}. Columns available: {list(df.columns)}")
        return []

    # 2. Validate metadata columns
    valid_metadata_cols = []
    for col in metadata_column_names:
        if col in df.columns:
            valid_metadata_cols.append(col)
        else:
            logger.warning(f"Metadata column '{col}' not found in {file_path}. Skipping.")

    if valid_metadata_cols:
        logger.info(f"Loading metadata from columns: {valid_metadata_cols}")

    # 3. Create Document objects
    for index, row in df.iterrows():
        text_content = row[text_col]

        # Skip rows where the text content is empty or NaN
        if pd.isna(text_content) or not str(text_content).strip():
            continue

        metadata_dict = {}
        for col in valid_metadata_cols:
            if col != text_col:  # Avoid duplicating text in metadata
                metadata_dict[col] = row[col]

        documents.append(Document(text=str(text_content), metadata=metadata_dict))

    logger.info(f"Loaded {len(documents)} documents with metadata from CSV file: {file_path}")
    return documents


def _load_documents_from_arrow(file_path: str,
                               text_column_name: Optional[str] = None,
                               metadata_column_names: Optional[List[str]] = None) -> List[Document]:
    """
    Loads LlamaIndex Documents from a single Arrow file.
    Uses 'text_column_name' for text and 'metadata_column_names' for metadata.
    """
    documents = []
    if metadata_column_names is None:
        metadata_column_names = []

    try:
        ds = datasets.Dataset.from_file(file_path)
    except Exception as e:
        logger.error(f"Failed to read Arrow file {file_path}: {e}")
        return []

    text_col = text_column_name

    # 1. Find text column
    if text_col and text_col in ds.column_names:
        logger.info(f"Using specified text column: '{text_col}'")
    else:
        if text_col:
            logger.warning(f"Specified text column '{text_col}' not found in Arrow file. Falling back to heuristic.")

        # Heuristic for arrow files
        text_col = next((c for c in ds.column_names if c in ['text', 'abstract', 'content']), None)

        if text_col:
            logger.info(f"Using heuristic text column: '{text_col}'")

    if not text_col:
        logger.error(f"No suitable text column found in {file_path}. Columns available: {ds.column_names}")
        return []

    # 2. Validate metadata columns
    valid_metadata_cols = []
    for col in metadata_column_names:
        if col in ds.column_names:
            valid_metadata_cols.append(col)
        else:
            logger.warning(f"Metadata column '{col}' not found in {file_path}. Skipping.")

    if valid_metadata_cols:
        logger.info(f"Loading metadata from columns: {valid_metadata_cols}")

    # 3. Create Document objects
    for ex in ds:
        text_content = ex.get(text_col)

        # Skip rows where the text content is empty
        if not text_content or not str(text_content).strip():
            continue

        metadata_dict = {}
        for col in valid_metadata_cols:
            if col != text_col:  # Avoid duplicating text in metadata
                metadata_dict[col] = ex.get(col)

        documents.append(Document(text=str(text_content), metadata=metadata_dict))

    logger.info(f"Loaded {len(documents)} documents with metadata from Arrow file: {file_path}")
    return documents


def load_local_dataset(local_dir: str) -> List[Document]:
    """
    Load a local dataset by reading 'dataset_dict.json'.
    - Iterates through all 'splits' defined in the json.
    - Loads all .arrow and .csv files from each split directory.
    - 'text_column': Specifies the column for document text.
    - 'metadata_columns': Specifies a list of columns for document metadata.
    Falls back to SimpleDirectoryReader if 'dataset_dict.json' is not found
    or if no documents are loaded from the specified splits.
    """
    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dict_path = os.path.join(PROJECT_PATH, "data", local_dir, "dataset_dict.json")
    data_path = os.path.join(PROJECT_PATH, "data", local_dir)
    all_documents = []

    try:
        if os.path.exists(dataset_dict_path):
            logger.info(f"Found dataset_dict.json in {local_dir}, loading data.")
            with open(dataset_dict_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)

            # Get the specified text and metadata columns from the JSON
            text_column = dataset_info.get("text_column")
            metadata_columns = dataset_info.get("metadata_columns", [])  # Default to empty list

            if text_column:
                logger.info(f"Using 'text_column' from JSON: {text_column}")
            if metadata_columns:
                logger.info(f"Using 'metadata_columns' from JSON: {metadata_columns}")

            splits = dataset_info.get("splits")
            if not splits:
                raise ValueError(f"No 'splits' key found in {dataset_dict_path}")

            logger.info(f"Processing splits: {splits}")

            for split_name in splits:
                logger.info(f"--- Loading split: {split_name} ---")
                split_dir = os.path.join(data_path, split_name)

                if not os.path.exists(split_dir):
                    logger.warning(
                        f"Split directory '{split_dir}' for split '{split_name}' does not exist. Skipping.")
                    continue

                # Find all .arrow or .csv files in the split directory
                data_files = [f for f in os.listdir(split_dir) if f.endswith(".arrow") or f.endswith(".csv")]

                if not data_files:
                    logger.warning(f"No .arrow or .csv files found in split directory: {split_dir}")
                    continue

                logger.info(f"Found {len(data_files)} data files in split '{split_name}'.")

                for data_file in data_files:
                    file_path = os.path.join(split_dir, data_file)
                    file_documents = []

                    if data_file.endswith(".arrow"):
                        logger.info(f"Loading documents from Arrow file: {data_file}")
                        file_documents = _load_documents_from_arrow(
                            file_path,
                            text_column_name=text_column,
                            metadata_column_names=metadata_columns
                        )
                    elif data_file.endswith(".csv"):
                        logger.info(f"Loading documents from CSV file: {data_file}")
                        file_documents = _load_documents_from_csv(
                            file_path,
                            text_column_name=text_column,
                            metadata_column_names=metadata_columns
                        )

                    all_documents.extend(file_documents)

            if all_documents:
                logger.info(f"✅ Loaded total {len(all_documents)} documents from all splits.")
                return all_documents
            else:
                logger.warning(
                    "dataset_dict.json was found, but no documents were loaded from any splits. Falling back.")

        logger.warning(f"Falling back to SimpleDirectoryReader for directory: {data_path}")
        reader = SimpleDirectoryReader(data_path)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents using SimpleDirectoryReader.")
        return documents

    except Exception as e:
        logger.error(f"Error loading local dataset from {local_dir}: {e}")
        raise