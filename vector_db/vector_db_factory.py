# vector_db_factory.py

"""
Vector Database Factory - Creates vector database instances based on type
"""

import logging
import os
from typing import Dict, Type
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utility.distance_metrics import DistanceMetric
from vector_db.simple_vector_db import SimpleVectorDB
from vector_db.chroma_vector_db import ChromaVectorDB
from vector_db.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VectorDBFactory:
    """
    Factory class for creating different types of vector databases.
    """

    # Registry of available vector database implementations
    _implementations: Dict[str, Type[VectorDBInterface]] = {
        'simple': SimpleVectorDB,
        'chroma': ChromaVectorDB,
    }

    @classmethod
    def create_vector_db(cls,
                         db_type: str,
                         source_path: str,
                         embedding_model: HuggingFaceEmbedding,
                         distance_metric: DistanceMetric = DistanceMetric.COSINE) -> VectorDBInterface:
        """
        Create a vector database instance based on the specified type.

        Args:
            db_type (str): Type of vector database ('simple', 'chroma')
            source_path (str): Path or identifier for the source
            embedding_model: Embedding model to use
            distance_metric: Distance metric to use for similarity search (default: COSINE)

        Returns:
            VectorDBInterface: The created vector database instance

        Raises:
            ValueError: If db_type is not supported

        Parameters
        ----------
        embedding_model
        source_path
        db_type
        distance_metric
        """
        if db_type not in cls._implementations:
            available_types = list(cls._implementations.keys())
            logger.error(f"Unsupported vector database type: {db_type}. Available types: {available_types}")
            raise ValueError(f"Unsupported vector database type: {db_type}. Available types: {available_types}")

        logger.info(f"Creating {db_type} vector database for source: {source_path}")
        print(f"Creating {db_type} vector database for source: {source_path}")

        # Get the implementation class and instantiate it
        implementation_class = cls._implementations[db_type]
        return implementation_class(source_path, embedding_model, distance_metric)

    @classmethod
    def register_implementation(cls, db_type: str, implementation_class: Type[VectorDBInterface]):
        """
        Register a new vector database implementation.

        Args:
            db_type (str): Name of the database type
            implementation_class (Type[VectorDBInterface]): Implementation class
        """
        if not issubclass(implementation_class, VectorDBInterface):
            raise TypeError(f"Implementation class must inherit from VectorDBInterface")

        cls._implementations[db_type] = implementation_class
        logger.info(f"Registered new vector database implementation: {db_type}")

    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available vector database types."""
        return list(cls._implementations.keys())

    @classmethod
    def get_implementation_info(cls, db_type: str) -> Dict:
        """
        Get information about a specific implementation.

        Args:
            db_type (str): Database type

        Returns:
            Dict: Information about the implementation
        """
        if db_type not in cls._implementations:
            raise ValueError(f"Unknown database type: {db_type}")

        impl_class = cls._implementations[db_type]
        return {
            "type": db_type,
            "class": impl_class.__name__,
            "module": impl_class.__module__,
            "docstring": impl_class.__doc__ or "No description available"
        }
