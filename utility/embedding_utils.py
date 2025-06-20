import string

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from logger import logger


def convert_text_into_vector(query: string, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a text query into a vector using the specified embedding model.

    Args:
        query (str): The text query to convert.
        embed_model (HuggingFaceEmbedding): The embedding model to use for conversion.

    Returns:
        np.ndarray: The resulting vector representation of the query.
    """
    logger.info(f"Converting text query into vector: {query[:30]}...")
    vector = embed_model.get_query_embedding(query)
    logger.info("Text query converted into vector successfully.")
    # return as nparray
    vector = np.array(vector, dtype=np.float32)
    return vector
