# modules/indexer.py - Fixed version
import os

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import HF_MODEL_NAME, USE_MIXED_PRECISION
from vector_db.chroma_index import build_chroma_vector_db
from vector_db.simple_index import build_simple_vector_db
from utility.logger import logger
from typing import Optional

PROJECT_PATH = os.path.abspath(__file__)


def load_vector_db(source: str = "local", source_path: Optional[str] = None, storing_method: str = "chroma") -> (
        VectorStoreIndex, HuggingFaceEmbedding):
    """
    Loads or creates a vector database for document retrieval with optimized embedding model caching.
    Simplified to avoid deprecated parameters and complex dtype handling.
    """
    logger.info(
        f"Loading vector database from source: {source}, source_path: {source_path}, storing_method: {storing_method}")

    # Simple approach - let HuggingFaceEmbedding handle everything
    try:
        embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
        logger.info(f"✅ Embedding model loaded successfully using simple approach")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise e

    # Cache the embedding model
    if not hasattr(load_vector_db, "_embed_model"):
        load_vector_db._embed_model = embedding_model

    # Build vector database
    if storing_method == "chroma":
        vector_db = build_chroma_vector_db(source=source, source_path=source_path, embedding_model=embedding_model)
        return vector_db, embedding_model

    elif storing_method == "llama_index":
        vector_db = build_simple_vector_db(source=source, source_path=source_path, embedding_model=embedding_model)
        return vector_db, embedding_model

    else:
        logger.error(f"Unsupported storing method: {storing_method}")
        raise ValueError(f"Unsupported storing method: {storing_method}")
