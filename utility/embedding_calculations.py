# embedding_calculations.py
import numpy as np
from logger import logger


def get_text_embedding(text, embed_model):
    """
    Generate a vector embedding for a given text using the embedding model.

    Args:
        text (str): The text to embed.
        embed_model: The embedding model.

    Returns:
        numpy.ndarray: Normalized vector embedding.
    """
    try:
        logger.info(f"Generating embedding for text: {text}")
        embedding = embed_model.get_text_embedding(text)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.warning("Embedding normalization resulted in division by zero. Returning original embedding.")
            return embedding
        normalized_embedding = embedding / norm
        logger.info(f"Generated normalized embedding: {normalized_embedding}")
        print(f"Normalized embedding for text '{text}': {normalized_embedding}")
        return normalized_embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text}': {e}")
        raise


def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector1 (numpy.ndarray): First vector.
        vector2 (numpy.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    try:
        logger.info(f"Calculating cosine similarity between vectors: {vector1}, {vector2}")
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("One or both vectors have zero magnitude. Returning similarity score of 0.0.")
            return 0.0
        similarity = np.dot(vector1, vector2) / (norm1 * norm2)
        logger.info(f"Calculated cosine similarity: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        raise
