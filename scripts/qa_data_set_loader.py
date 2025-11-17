import json
from pathlib import Path

from datasets import load_dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar

from configurations.config import NQ_SAMPLE_SIZE, ACTIVE_QA_DATASET, QA_DATASETS
from utility.logger import logger

PROJECT_PATH = Path(__file__).resolve().parents[1]


def getQACollectionsPath(collection_name: str) -> Path:
    return PROJECT_PATH / "QACollection" / collection_name


def get_dataset_dirname(name: str, config: str | None = None) -> str:
    dirname = name.replace("/", "_")
    if config:
        dirname += f"_{config}"
    return dirname


def load_qa_queries(sample_size: int = NQ_SAMPLE_SIZE) -> list[dict]:
    """
    Loads QA entries from local JSONL dataset.
    Returns full dict entries (question + pubid + all other fields).
    """

    if ACTIVE_QA_DATASET not in QA_DATASETS:
        raise ValueError(f"Dataset '{ACTIVE_QA_DATASET}' not found in QA_DATASETS config.")

    dataset_info = QA_DATASETS[ACTIVE_QA_DATASET]
    config = dataset_info.get("config", None)

    dataset_dirname = get_dataset_dirname(ACTIVE_QA_DATASET, config)

    optional_json_files = ["train.json", "validation.json", "test.json"]

    existing_file = None
    for file_name in optional_json_files:
        dataset_path = getQACollectionsPath(dataset_dirname) / file_name
        if dataset_path.exists():
            existing_file = dataset_path
            break

    if not existing_file:
        raise FileNotFoundError(f"Dataset file not found in : {optional_json_files}")

    entries = []
    with open(existing_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_size != -1 and i >= sample_size:
                break
            try:
                data = json.loads(line)
                entries.append(data)
            except Exception as e:
                logger.warning(f"Skipping malformed line {i}: {e}")

    return entries


def download_qa_dataset():
    """
    Download a QA dataset from HuggingFace and save it under 'qa_collection' in the project root.

    The dataset is selected based on ACTIVE_QA_DATASET and QA_DATASETS from config.py.
    Dataset is saved in JSON format under: <project_root>/qa_collection/<dataset_name>[_<config_name>]
    """
    disable_progress_bar()

    try:
        if not PROJECT_PATH.exists():
            raise FileNotFoundError(f"Project root not found: {PROJECT_PATH}")

        # Step 1: Validate dataset name from config
        if ACTIVE_QA_DATASET not in QA_DATASETS:
            raise ValueError(f"Dataset '{ACTIVE_QA_DATASET}' not found in QA_DATASETS config.")

        dataset_info = QA_DATASETS[ACTIVE_QA_DATASET]
        config = dataset_info.get("config", None)

        # Step 2: Prepare safe directory name (replace '/' to '_')
        dataset_dirname = get_dataset_dirname(ACTIVE_QA_DATASET, config)

        save_dir = getQACollectionsPath(dataset_dirname)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦 Downloading QA dataset: '{ACTIVE_QA_DATASET}' (config: {config})")
        print(f"💾 Saving to: {save_dir}")

        # Step 3: Download from HuggingFace
        if config:
            dataset: DatasetDict = load_dataset(ACTIVE_QA_DATASET, config)
        else:
            dataset: DatasetDict = load_dataset(ACTIVE_QA_DATASET)

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
