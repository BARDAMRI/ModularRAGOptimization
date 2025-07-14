# modules/indexer.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import HF_MODEL_NAME
from vector_db.vector_db_factory import VectorDBFactory
from vector_db.vector_db_interface import VectorDBInterface
from utility.logger import logger
from utility.device_utils import get_optimal_device
from typing import Tuple


def load_vector_db(source_path: str = "local_data_dir",
                   storing_method: str = "chroma") -> Tuple[VectorDBInterface, HuggingFaceEmbedding]:
    """
    Loads or creates a vector database for document retrieval with optimized embedding model caching.
    Now returns a VectorDBInterface instead of raw VectorStoreIndex.

    Args:
        source_path (str): Path to the data source (local directory, URL, or HF dataset identifier)
        storing_method (str): Storage method to use ('chroma', 'simple', 'llama_index')

    Returns:
        Tuple[VectorDBInterface, HuggingFaceEmbedding]: Vector database interface and embedding model
    """
    logger.info(f"Loading vector database for source_path: '{source_path}', storing_method: '{storing_method}'")

    # Get or create cached embedding model
    embedding_model = _get_cached_embedding_model()

    # Map storing_method to factory db_type for backward compatibility
    db_type_mapping = {
        "chroma": "chroma",
        "llama_index": "simple",  # Keep backward compatibility
        "simple": "simple"
    }

    if storing_method not in db_type_mapping:
        available_methods = list(db_type_mapping.keys())
        logger.error(f"Unsupported storing method: {storing_method}. Available methods: {available_methods}")
        raise ValueError(f"Unsupported storing method: {storing_method}. Available methods: {available_methods}")

    db_type = db_type_mapping[storing_method]

    try:
        # Use the factory to create the vector database interface
        vector_db = VectorDBFactory.create_vector_db(
            db_type=db_type,
            source_path=source_path,
            embedding_model=embedding_model
        )

        logger.info(f"✅ Vector database interface created successfully using {storing_method} method")
        logger.info(f"Database stats: {vector_db.get_stats()}")

        return vector_db, embedding_model

    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        raise e


def _get_cached_embedding_model() -> HuggingFaceEmbedding:
    """
    Get or create a cached embedding model to avoid reloading.

    Returns:
        HuggingFaceEmbedding: The cached embedding model
    """
    if not hasattr(load_vector_db, "_embed_model"):
        try:
            logger.info(f"Loading HuggingFaceEmbedding with model_name: {HF_MODEL_NAME}")
            device = get_optimal_device()

            embedding_model = HuggingFaceEmbedding(
                model_name=HF_MODEL_NAME,
                device=str(device)
            )

            load_vector_db._embed_model = embedding_model
            logger.info(f"✅ Embedding model loaded and cached successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e
    else:
        logger.info("Using cached embedding model")

    return load_vector_db._embed_model


def get_available_storing_methods() -> list:
    """
    Get list of available storing methods.

    Returns:
        list: Available storing methods
    """
    return ["chroma", "simple", "llama_index"]


def get_vector_db_info(storing_method: str) -> dict:
    """
    Get information about a specific vector database implementation.

    Args:
        storing_method (str): The storing method name

    Returns:
        dict: Information about the implementation
    """
    db_type_mapping = {
        "chroma": "chroma",
        "llama_index": "simple",
        "simple": "simple"
    }

    if storing_method in db_type_mapping:
        db_type = db_type_mapping[storing_method]
        return VectorDBFactory.get_implementation_info(db_type)
    else:
        raise ValueError(f"Unknown storing method: {storing_method}")


# Utility functions for debugging and testing
def test_vector_db_creation(source_path: str = "test_data", storing_method: str = "simple"):
    """
    Test function to verify vector database creation works correctly.

    Args:
        source_path (str): Path to test data
        storing_method (str): Method to test
    """
    try:
        logger.info(f"Testing vector DB creation with method: {storing_method}")
        vector_db, embedding_model = load_vector_db(source_path, storing_method)

        # Test basic functionality
        stats = vector_db.get_stats()
        logger.info(f"Test successful! Stats: {stats}")

        # Test retrieval if data exists
        try:
            results = vector_db.retrieve("test query", top_k=3)
            logger.info(f"Retrieval test: Found {len(results)} results")
        except Exception as e:
            logger.warning(f"Retrieval test failed (this is normal if no data): {e}")

        return True

    except Exception as e:
        logger.error(f"Vector DB creation test failed: {e}")
        return False


def get_embedding_model_info() -> dict:
    return {
        "model_name": HF_MODEL_NAME,
        "device": str(get_optimal_device()),
        "is_cached": False,
        "model_type": "HuggingFaceEmbedding"
    }
