# utility/embedding_utils.py
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import performance monitoring and caching
try:
    from utility.cache import cache_embedding
    from utility.performance import monitor_performance

    CACHE_AVAILABLE = True
    PERFORMANCE_AVAILABLE = True
except ImportError as e:
    CACHE_AVAILABLE = False
    PERFORMANCE_AVAILABLE = False


    def cache_embedding(model_name):
        def decorator(func):
            return func

        return decorator


    def monitor_performance(name):
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield

        return dummy_context()


@cache_embedding("sentence-transformers/all-MiniLM-L6-v2")
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


def calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector1 (np.ndarray): First vector
        vector2 (np.ndarray): Second vector

    Returns:
        float: Cosine similarity score
    """
    with monitor_performance("cosine_similarity_calculation"):
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return np.dot(vector1, vector2) / (norm1 * norm2)


def batch_embed_texts(texts: list, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Embed multiple texts efficiently.

    Args:
        texts (list): List of texts to embed
        embed_model (HuggingFaceEmbedding): The embedding model

    Returns:
        np.ndarray: Array of embeddings
    """
    with monitor_performance("batch_embedding"):
        embeddings = []
        for text in texts:
            embedding = get_text_embedding(text, embed_model)
            embeddings.append(embedding)
        return np.array(embeddings)
