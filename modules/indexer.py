# modules/indexer.py - Updated to use VectorDBFactory
import os
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import HF_MODEL_NAME, USE_MIXED_PRECISION
from vector_db.vector_db_factory import VectorDBFactory
from typing import Optional, Tuple

PROJECT_PATH = os.path.abspath(__file__)


def load_vector_db(logger,
                   source: str = "local",
                   source_path: Optional[str] = None,
                   storing_method: str = "chroma",
                   ) -> Tuple[VectorStoreIndex, HuggingFaceEmbedding]:
    """
    Loads or creates a vector database for document retrieval with optimized embedding model caching.
    Now uses VectorDBFactory for better organization and extensibility.

    Args:
        source (str): 'local' or 'url'
        source_path (Optional[str]): Path or identifier for the source
        storing_method (str): Type of vector database ('chroma', 'simple', etc.)

    Returns:
        Tuple[VectorStoreIndex, HuggingFaceEmbedding]: The vector database and embedding model
    """
    logger.info(
        f"Loading vector database from source: {source}, source_path: {source_path}, storing_method: {storing_method}")

    # Get or create cached embedding model
    embedding_model = _get_cached_embedding_model(logger=logger)

    # Map storing_method to factory db_type
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
        # Use the factory to create the vector database
        vector_db = VectorDBFactory.create_vector_db(
            db_type=db_type,
            source=source,
            source_path=source_path,
            embedding_model=embedding_model,
            logger=logger
        )

        logger.info(f"✅ Vector database created successfully using {storing_method} method")
        return vector_db, embedding_model

    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        raise e


def _get_cached_embedding_model(logger) -> HuggingFaceEmbedding:
    """
    Get or create a cached embedding model to avoid reloading.

    Returns:
        HuggingFaceEmbedding: The cached embedding model
    """
    # Check if we already have a cached model
    if not hasattr(load_vector_db, "_embed_model"):
        try:
            logger.info(f"Loading embedding model: {HF_MODEL_NAME}")
            embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)

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
    return ["chroma", "llama_index", "simple"]


def clear_embedding_cache(logger):
    """Clear the cached embedding model (useful for testing or memory management)"""
    if hasattr(load_vector_db, "_embed_model"):
        delattr(load_vector_db, "_embed_model")
        logger.info("Embedding model cache cleared")
