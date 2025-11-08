# utility/similarity_calculator.py
import numpy as np
from typing import Callable, Union
from enum import Enum


class SimilarityMethod(Enum):
    """Enumeration of available similarity methods."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class SimilarityCalculator:
    """High-performance similarity calculator with extensible architecture."""

    def __init__(self):
        # Core methods with optimized implementations
        self._batch_methods = {
            SimilarityMethod.COSINE: self._cosine_batch,
            SimilarityMethod.DOT_PRODUCT: self._dot_batch,
            SimilarityMethod.EUCLIDEAN: self._euclidean_batch,
        }

        self._pairwise_methods = {
            SimilarityMethod.COSINE: self._cosine_pairwise,
            SimilarityMethod.DOT_PRODUCT: self._dot_pairwise,
            SimilarityMethod.EUCLIDEAN: self._euclidean_pairwise,
        }

        # For custom methods
        self._custom_methods = {}

    def calculate_batch_similarity(
            self,
            query_vector: np.ndarray,
            document_embeddings: np.ndarray,
            method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
    ) -> np.ndarray:
        """
        High-performance batch similarity calculation.

        Args:
            query_vector: 1D query vector
            document_embeddings: 2D array (n_docs, embedding_dim)
            method: Similarity method

        Returns:
            np.ndarray: Similarity scores for all documents
        """
        if callable(method):
            # Custom function - use pairwise calculation
            return np.array([method(query_vector, doc) for doc in document_embeddings])

        if isinstance(method, str):
            # Check custom methods first
            if method in self._custom_methods:
                return np.array([self._custom_methods[method](query_vector, doc) for doc in document_embeddings])
            method = SimilarityMethod(method)

        return self._batch_methods[method](query_vector, document_embeddings)

    def calculate_pairwise_similarity(
            self,
            vector1: np.ndarray,
            vector2: np.ndarray,
            method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
    ) -> float:
        """
        Calculate similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector
            method: Similarity method

        Returns:
            float: Similarity score
        """
        if callable(method):
            return method(vector1, vector2)

        if isinstance(method, str):
            if method in self._custom_methods:
                return self._custom_methods[method](vector1, vector2)
            method = SimilarityMethod(method)

        return self._pairwise_methods[method](vector1, vector2)

    def add_method(self, name: str, pairwise_func: Callable[[np.ndarray, np.ndarray], float]):
        """
        Add a custom similarity method.

        Args:
            name: Method name
            pairwise_func: Function that takes two vectors and returns similarity
        """
        self._custom_methods[name] = pairwise_func

    # Optimized batch methods
    def _cosine_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity."""
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0.0:
            return np.zeros(len(document_embeddings))

        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        zero_mask = doc_norms == 0.0
        doc_norms = np.where(zero_mask, 1.0, doc_norms)  # Avoid division by zero

        similarities = np.dot(document_embeddings, query_vector) / (doc_norms * query_norm)
        similarities[zero_mask] = 0.0

        return similarities

    def _dot_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized dot product."""
        return np.dot(document_embeddings, query_vector)

    def _euclidean_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance converted to similarity."""
        distances = np.linalg.norm(document_embeddings - query_vector, axis=1)
        return 1.0 / (1.0 + distances)

    # Pairwise methods
    def _cosine_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise cosine similarity."""
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return np.dot(vector1, vector2) / (norm1 * norm2)

    def _dot_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise dot product."""
        return np.dot(vector1, vector2)

    def _euclidean_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise Euclidean distance to similarity."""
        distance = np.linalg.norm(vector1 - vector2)
        return 1.0 / (1.0 + distance)


# Global instance
_calculator = SimilarityCalculator()


def calculate_similarities(
        query_vector: np.ndarray,
        document_embeddings: np.ndarray,
        method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> np.ndarray:
    """
    Main function for batch similarity calculation.

    Args:
        query_vector: Query vector (1D)
        document_embeddings: Document embeddings (2D)
        method: Similarity method

    Returns:
        np.ndarray: Similarity scores
    """
    return _calculator.calculate_batch_similarity(query_vector, document_embeddings, method)


def calculate_similarity(
        vector1: np.ndarray,
        vector2: np.ndarray,
        method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> float:
    """
    Main function for pairwise similarity calculation.

    Args:
        vector1: First vector
        vector2: Second vector
        method: Similarity method

    Returns:
        float: Similarity score
    """
    return _calculator.calculate_pairwise_similarity(vector1, vector2, method)


def add_similarity_method(name: str, func: Callable[[np.ndarray, np.ndarray], float]):
    """
    Add a custom similarity method globally.

    Args:
        name: Method name
        func: Pairwise similarity function
    """
    _calculator.add_method(name, func)
