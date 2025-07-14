"""
Simple Vector Database Implementation
"""

import os
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url
from utility.logger import logger
from vector_db.vector_db_interface import VectorDBInterface

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_storage_directory(self, parsed_name: str) -> str:
    """
    Construct and return the consistent storage directory path based on parsed name.

    Args:
        parsed_name (str): Cleaned version of source path name

    Returns:
        str: Full storage directory path
    """
    return os.path.join(PROJECT_PATH, "storage", self.db_type, parsed_name.replace(":", "_"))


class SimpleVectorDB(VectorDBInterface):
    """
    Simple vector database implementation using LlamaIndex's default storage.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding):
        """
        Initialize Simple vector database.

        Args:
            source_path (str): Path to the data source
            embedding_model: Embedding model to use
        """
        self.storage_dir = None
        super().__init__(source_path, embedding_model)

    def _initialize(self) -> None:
        """Initialize the Simple vector database."""
        logger.info(f"Initializing Simple vector database for source: {self.source_path}")

        # Parse source and prepare data
        source_type, parsed_name = parse_source_path(self.source_path)
        data_dir = self._prepare_data_directory(source_type, parsed_name)

        # Setup storage directory
        self.storage_dir = _get_storage_directory(self.db_type, parsed_name)
        os.makedirs(self.storage_dir, exist_ok=True)

        # Load existing index or create new one
        if self._index_exists():
            logger.info(f"Loading existing Simple vector database from {self.storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.vector_db = load_index_from_storage(storage_context, embed_model=self.embedding_model)
        else:
            logger.info(f"Creating new Simple vector database from {data_dir}")
            documents = self._load_documents(data_dir)
            self.vector_db = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embedding_model
            )
            self.persist()
            logger.info(f"Indexed {len(documents)} documents into Simple vector DB")

        # Verify the vector_db is properly set
        if not isinstance(self.vector_db, VectorStoreIndex):
            raise RuntimeError(f"Expected VectorStoreIndex, got {type(self.vector_db)}")

        logger.info(f"Simple VectorDB initialized successfully with {type(self.vector_db).__name__}")

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve documents using Simple vector store.

        Args:
            query (str): Query string
            top_k (int): Number of top results to return

        Returns:
            List[NodeWithScore]: Retrieved documents with scores
        """
        logger.info(f"Simple retrieval for query: '{query}' (top_k={top_k})")

        retriever = self.vector_db.as_retriever(similarity_top_k=top_k)
        nodes_with_scores = retriever.retrieve(query)

        # Simple store specific post-processing could go here
        # For example: custom scoring, filtering, etc.

        logger.info(f"Simple vector DB retrieved {len(nodes_with_scores)} documents")
        return nodes_with_scores

    def semantic_search(self, query: str, top_k: int = 5,
                        similarity_threshold: float = 0.0) -> List[NodeWithScore]:
        """
        Simple store specific semantic search with threshold filtering.

        Args:
            query (str): Query string
            top_k (int): Number of results
            similarity_threshold (float): Minimum similarity score

        Returns:
            List[NodeWithScore]: Retrieved documents above threshold
        """
        nodes_with_scores = self.retrieve(query, top_k)

        # Filter by similarity threshold
        if similarity_threshold > 0.0:
            filtered_nodes = [
                node for node in nodes_with_scores
                if node.score >= similarity_threshold
            ]
            logger.info(
                f"Filtered {len(nodes_with_scores)} -> {len(filtered_nodes)} nodes by threshold {similarity_threshold}")
            return filtered_nodes

        return nodes_with_scores

    def add_documents(self, documents: List[Any]) -> None:
        """
        Add documents to the Simple vector database.

        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to Simple vector DB")
        self.vector_db.insert_nodes(documents)
        self.persist()
        logger.info("Documents added to Simple vector DB successfully")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Simple database statistics.

        Returns:
            Dict containing Simple vector DB stats
        """
        stats = {
            "db_type": self.db_type,
            "storage_dir": self.storage_dir,
            "embedding_model": self.embedding_model.model_name,
            "index_exists": self._index_exists()
        }

        # Try to get document count if possible
        try:
            if hasattr(self.vector_db, 'docstore') and hasattr(self.vector_db.docstore, 'docs'):
                stats["document_count"] = len(self.vector_db.docstore.docs)
        except Exception as e:
            logger.debug(f"Could not get document count: {e}")
            stats["document_count"] = "unknown"

        return stats

    def persist(self) -> None:
        """
        Persist the Simple vector database to storage.
        """
        logger.info(f"Persisting Simple vector database to {self.storage_dir}")
        self.vector_db.storage_context.persist(persist_dir=self.storage_dir)

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "simple"

    def _index_exists(self) -> bool:
        """Check if an index already exists in the storage directory."""
        return (os.path.exists(self.storage_dir) and
                os.path.exists(os.path.join(self.storage_dir, "docstore.json")))

    def _prepare_data_directory(self, source_type: str, parsed_name: str) -> str:
        """Prepare and return the data directory path."""
        if source_type == "url":
            data_dir = os.path.join(PROJECT_PATH, "data", "url_downloads", parsed_name)
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                logger.info(f"Downloading data from URL into {data_dir}")
                download_and_save_from_url(self.source_path, data_dir)
        elif source_type == "hf":
            data_dir = os.path.join(PROJECT_PATH, "data", "hf_downloads", parsed_name.replace(":", "_"))
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                logger.info(f"Downloading data from Hugging Face into {data_dir}")
                # Parse dataset_name and config from source_path
                if ":" in self.source_path:
                    dataset_name, config = self.source_path.split(":", 1)
                else:
                    dataset_name, config = self.source_path, None
                download_and_save_from_hf(dataset_name, config, data_dir, max_docs=1000)
        else:  # local source
            data_dir = self.source_path

        return data_dir

    def _load_documents(self, data_dir: str):
        """Load documents from the data directory."""
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            logger.error(f"No documents found in {data_dir}")
            raise FileNotFoundError(f"No documents found in {data_dir}")

        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            logger.warning(f"No documents loaded from {data_dir}")
            return []

        return documents
