"""
Base interface for vector database implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class VectorDBInterface(ABC):
    """
    Abstract base class for all vector database implementations.
    Provides a unified interface for different vector store types.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding):
        """
        Initialize the vector database.

        Args:
            source_path (str): Path to the data source
            embedding_model: Embedding model to use
        """
        self.source_path = source_path
        self.embedding_model = embedding_model
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

    def as_retriever(self, **kwargs):
        """
        Get a retriever for this vector database.
        Default implementation - delegates to underlying VectorStoreIndex.
        """
        if self.vector_db is None:
            raise RuntimeError("Vector database not initialized")

        if hasattr(self.vector_db, 'as_retriever'):
            return self.vector_db.as_retriever(**kwargs)
        else:
            raise NotImplementedError(f"Retriever not available for this vector DB type: {type(self.vector_db)}")
