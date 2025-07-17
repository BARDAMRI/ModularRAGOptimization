"""
Base interface for vector database implementations
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import numpy as np
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utility.distance_metrics import DistanceMetric

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_distance_metric(metric: Union[str, DistanceMetric]) -> str:
    if isinstance(metric, DistanceMetric):
        return metric.value.lower()
    return metric.lower()


class VectorDBInterface(ABC):
    """
    Abstract base class for all vector database implementations.
    Provides a unified interface for different vector store types.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding, distance_metric: DistanceMetric):
        """
        Initialize the vector database.

        Args:
            source_path (str): Path to the data source
            embedding_model: Embedding model to use
        """
        self.source_path = source_path
        self.embedding_model = embedding_model
        self.distance_metric = _normalize_distance_metric(distance_metric)
        self.vector_db = None
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the specific vector database implementation.
        Should set self.vector_db to the appropriate vector store.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query (str): Query string
            top_k (int): Number of top results to return

        Returns:
            List[NodeWithScore]: Retrieved documents with scores
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Any]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents to add
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dict containing stats like document count, storage size, etc.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persist the vector database to storage.
        """
        pass

    @property
    @abstractmethod
    def db_type(self) -> str:
        """
        Return the type of vector database.

        Returns:
            str: Database type identifier
        """
        pass

    def as_query_engine(self, **kwargs):
        """
        Get a query engine for this vector database.
        Default implementation - delegates to underlying VectorStoreIndex.
        """
        if self.vector_db is None:
            raise RuntimeError("Vector database not initialized")

        if hasattr(self.vector_db, 'as_query_engine'):
            return self.vector_db.as_query_engine(**kwargs)
        else:
            raise NotImplementedError(f"Query engine not available for this vector DB type: {type(self.vector_db)}")

    def euclidean_distance(self, x, y):
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x - y)

    def _get_storage_directory(self, parsed_name: str) -> str:
        """
        Construct and return the consistent storage directory path
        based on parsed name and distance metric.

        Args:
            parsed_name (str): Cleaned version of source path name

        Returns:
            str: Full storage directory path
        """
        return os.path.join(
            PROJECT_PATH,
            "storage",
            self.db_type,
            parsed_name.replace(":", "_"),
            self.distance_metric
        )
