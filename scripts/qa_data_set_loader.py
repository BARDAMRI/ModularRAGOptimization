import json
from pathlib import Path
from datasets import load_dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from configurations.config import QA_DATASET_NAME, NQ_SAMPLE_SIZE
from utility.logger import logger

PROJECT_PATH = Path(__file__).resolve().parents[1]


def getQACollectionsPath(collection_name: str) -> Path:
    return PROJECT_PATH / "QACollection" / collection_name


def load_qa_queries(sample_size: int = NQ_SAMPLE_SIZE) -> list[str]:
    """
    Loads a list of QA queries from the local QA dataset.

    Args:
        sample_size (int): Number of queries to load.

    Returns:
        List[str]: List of questions loaded from the dataset.
    """
    dataset_dirname = QA_DATASET_NAME.replace("/", "_")
    dataset_path = getQACollectionsPath(dataset_dirname) / "train.json"

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    queries = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                data = json.loads(line)
                question = data.get("question")
                if question:
                    queries.append(question)
            except Exception as e:
                logger.warning(f"Skipping malformed line {i}: {e}")

    return queries


def download_qa_dataset():
    """
    Download a QA dataset from HuggingFace and save it under 'qa_collection' in the project root.

    The dataset name is taken from the config (QA_DATASET_NAME).
    Dataset is saved in JSON format under: <project_root>/qa_collection/<dataset_name>
    """
    disable_progress_bar()

    try:
        if not PROJECT_PATH.exists():
            raise FileNotFoundError(f"Project root not found: {PROJECT_PATH}")

        # Step 2: Validate dataset name
        if not QA_DATASET_NAME or not isinstance(QA_DATASET_NAME, str):
            raise ValueError("QA_DATASET_NAME in config is invalid or missing.")
        if any(char in QA_DATASET_NAME for char in ['\\', ':', '*', '?', '"', '<', '>', '|']):
            raise ValueError(f"Invalid characters in dataset name: '{QA_DATASET_NAME}'")

        dataset_dirname = QA_DATASET_NAME.replace("/", "_")
        save_dir = getQACollectionsPath(dataset_dirname)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦 Downloading QA dataset: '{QA_DATASET_NAME}'")
        print(f"💾 Saving to: {save_dir}")

        # Step 3: Download from HuggingFace
        dataset: DatasetDict = load_dataset(QA_DATASET_NAME)

        # Step 4: Save each split to JSONL
        for split_name, split_data in dataset.items():
            output_file = save_dir / f"{split_name}.json"
            split_data.to_json(str(output_file), orient="records", lines=True)
            print(f"✅ Saved split: {output_file}")

        print("🎉 Dataset downloaded and saved successfully.")

    except FileNotFoundError as fnf:
        logger.error(f"❌ Directory error: {fnf}")
        print(f"❌ Could not resolve directory: {fnf}")
    except ValueError as ve:
        logger.error(f"❌ Invalid dataset configuration: {ve}")
        print(f"❌ Configuration error: {ve}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"❌ Download failed: {e}")
