# modules/indexer.py
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import HF_MODEL_NAME, USE_MIXED_PRECISION
from vector_db.vector_db_factory import VectorDBFactory
from vector_db.vector_db_interface import VectorDBInterface
from utility.logger import logger
from utility.device_utils import get_optimal_device
from typing import Tuple
from sentence_transformers import SentenceTransformer


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
    # Check if we already have a cached model
    if not hasattr(load_vector_db, "_embed_model"):
        try:
            logger.info(f"Loading embedding model: {HF_MODEL_NAME}")
            device = get_optimal_device()

            # Create SentenceTransformer with device-specific settings
            if device.type == "mps":
                logger.info(f"Loading SentenceTransformer for MPS with float32")
                sbert_model = SentenceTransformer(HF_MODEL_NAME, device=str(device))
                # Convert to float32 for MPS compatibility
                sbert_model = sbert_model.float()
            elif device.type == "cuda":
                logger.info(f"Loading SentenceTransformer for CUDA")
                sbert_model = SentenceTransformer(HF_MODEL_NAME, device=str(device))
                if USE_MIXED_PRECISION:
                    logger.info("Converting model to float16 for mixed precision")
                    sbert_model = sbert_model.half()
                else:
                    logger.info("Using float32 for CUDA")
                    sbert_model = sbert_model.float()
            else:  # CPU
                logger.info(f"Loading SentenceTransformer for CPU with float32")
                sbert_model = SentenceTransformer(HF_MODEL_NAME, device=str(device))
                sbert_model = sbert_model.float()

            # Create HuggingFaceEmbedding with pre-loaded model
            embedding_model = HuggingFaceEmbedding(
                model=sbert_model,
                model_name=HF_MODEL_NAME,
                device=str(device)
            )

            # Cache the embedding model
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


def clear_embedding_cache():
    """Clear the cached embedding model (useful for testing or memory management)"""
    if hasattr(load_vector_db, "_embed_model"):
        delattr(load_vector_db, "_embed_model")
        logger.info("Embedding model cache cleared")


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
    """
    Get information about the current embedding model.

    Returns:
        dict: Embedding model information
    """
    if hasattr(load_vector_db, "_embed_model"):
        model = load_vector_db._embed_model
        return {
            "model_name": model.model_name,
            "device": model.device,
            "is_cached": True,
            "model_type": type(model).__name__
        }
    else:
        return {
            "model_name": HF_MODEL_NAME,
            "device": str(get_optimal_device()),
            "is_cached": False,
            "model_type": "Not loaded"
        }
