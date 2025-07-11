"""
Vector Database Factory - Main entry point for creating different types of vector databases
"""

from utility.logger import logger as global_logger
from typing import Dict, Callable
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from vector_db.chroma_index import build_chroma_vector_db
from vector_db.simple_index import build_simple_vector_db


class VectorDBFactory:
    """Factory class for creating different types of vector databases"""

    # Registry of available vector database builders
    _builders: Dict[str, Callable] = {
        'simple': build_simple_vector_db,
        'chroma': build_chroma_vector_db,
    }

    @classmethod
    def create_vector_db(cls,
                         db_type: str,
                         source: str,
                         source_path: str,
                         embedding_model: HuggingFaceEmbedding,
                         logger) -> VectorStoreIndex:
        """
        Create a vector database based on the specified type.

        Args:
            db_type (str): Type of vector database ('simple', 'chroma')
            source (str): 'local' or 'url'
            source_path (str): Path or identifier for the source
            embedding_model: Embedding model to use

        Returns:
            VectorStoreIndex: The created vector store index

        Raises:
            ValueError: If db_type is not supported
        """
        if db_type not in cls._builders:
            available_types = list(cls._builders.keys())
            logger.error(f"Unsupported vector database type: {db_type}. Available types: {available_types}")
            raise ValueError(f"Unsupported vector database type: {db_type}. Available types: {available_types}")

        logger.info(f"Creating {db_type} vector database with source: {source}")
        builder_func = cls._builders[db_type]
        return builder_func(source, source_path, embedding_model, logger)

    @classmethod
    def register_builder(cls, db_type: str, builder_func: Callable):
        """
        Register a new vector database builder.

        Args:
            db_type (str): Name of the database type
            builder_func (Callable): Function that builds the database
        """
        cls._builders[db_type] = builder_func
        global_logger.info(f"Registered new vector database builder: {db_type}")

    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available vector database types"""
        return list(cls._builders.keys())
