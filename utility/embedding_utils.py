import string

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# from logger import logger


def get_document_vector(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a document's text into a vector embedding using the specified embedding model.

    Args:
        text (str): The input document text.
        embed_model (HuggingFaceEmbedding): The embedding model used for generating embeddings.

    Returns:
        np.ndarray: The vector embedding of the document text.
    """
    # logger.info(f"Generating document vector for text: {text[:30]}...")
    vector = embed_model.get_text_embedding(text)
    # logger.info("Document vector generated successfully.")
    return np.array(vector, dtype=np.float32)


def get_query_vector(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a query's text into a vector embedding using the specified embedding model.

    Args:
        text (str): The input query text.
        embed_model (HuggingFaceEmbedding): The embedding model used for generating embeddings.

    Returns:
        np.ndarray: The vector embedding of the query text.
    """
    # logger.info(f"Generating query vector for text: {text[:30]}...")
    vector = embed_model.get_query_embedding(text)
    # logger.info("Query vector generated successfully.")
    return np.array(vector, dtype=np.float32)
