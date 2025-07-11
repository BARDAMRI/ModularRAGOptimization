def parse_source_path(source_path: str) -> Tuple[str, str]:
    """
    Parses the source path to determine its type and extract relevant information.

    Args:
        source_path (str): The source path to parse.

    Returns:
        Tuple[str, str]: A tuple containing the source type ('url' or 'hf') and the parsed corpus or dataset configuration.

    Raises:
        ValueError: If the source path format is unsupported.
    """
    logger.info(f"Parsing source path: {source_path}")
    if source_path.startswith("http://") or source_path.startswith("https://"):
        corpus_name = source_path.split("/")[-1]
        logger.info(f"Source type identified as URL with corpus name: {corpus_name}")
        return "url", corpus_name
    elif ":" in source_path:
        dataset_config = source_path.replace(":", "_")
        logger.info(f"Source type identified as Hugging Face dataset with config: {dataset_config}")
        return "hf", dataset_config
    else:
        logger.error(f"Unsupported source path format: {source_path}")
        raise ValueError(f"Unsupported source path format: {source_path}")


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


def download_and_save_from_hf(dataset_name: str, config: str, target_dir: str, max_docs: int = 1000) -> None:
    """
    Downloads a dataset from Hugging Face and saves it locally as text files in batches.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        config (str): Configuration for the dataset.
        target_dir (str): Directory to save the downloaded documents.
        max_docs (int): Maximum number of documents to download.

    Returns:
        None
    """
    logger.info(f"Downloading dataset '{dataset_name}' with config '{config}' from Hugging Face...")
    dataset = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
    os.makedirs(target_dir, exist_ok=True)

    batch_size = 100
    batch = []
    for i, item in enumerate(dataset):
        if i >= max_docs:
            break
        text = item["text"].strip()
        if text:
            batch.append((i, text))
        if len(batch) == batch_size or i == max_docs - 1:
            for doc_id, doc_text in batch:
                with open(os.path.join(target_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(doc_text)
            batch.clear()
    logger.info(f"Saved {min(max_docs, len(dataset))} documents to {target_dir}")
