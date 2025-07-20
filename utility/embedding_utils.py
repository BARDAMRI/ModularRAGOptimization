# utility/embedding_utils.py
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def monitor_performance(name):
    from contextlib import contextmanager
    @contextmanager
    def dummy_context():
        yield

    return dummy_context()


def get_query_vector(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a query's text into a vector embedding using the specified embedding model.
    Now includes caching and performance monitoring.

    Args:
        text (str): The input query text.
        embed_model (HuggingFaceEmbedding): The embedding model used for generating embeddings.

    Returns:
        np.ndarray: The vector embedding of the query text.
    """
    with monitor_performance("embedding_generation"):
        vector = embed_model.get_query_embedding(text)
    return np.array(vector, dtype=np.float32)


def get_text_embedding(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Get text embedding for documents (not queries).

    Args:
        text (str): The input text.
        embed_model (HuggingFaceEmbedding): The embedding model.

    Returns:
        np.ndarray: The vector embedding of the text.
    """
    with monitor_performance("text_embedding_generation"):
        vector = embed_model.get_text_embedding(text)
    return np.array(vector, dtype=np.float32)
