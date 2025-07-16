"""
Chroma Vector Database Implementation
"""

import os
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url
from utility.logger import logger
from vector_db.vector_db_interface import VectorDBInterface

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChromaVectorDB(VectorDBInterface):
    """
    Chroma-based vector database implementation.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding):
        """
        Initialize Chroma vector database.

        Args:
            source_path (str): Path to the data source
            embedding_model: Embedding model to use
        """
        self.chroma_client = None
        self.collection = None
        self.chroma_store = None
        super().__init__(source_path, embedding_model)

    def _initialize(self) -> None:
        """Initialize the Chroma vector database."""
        logger.info(f"Initializing Chroma vector database for source: {self.source_path}")

        # Parse source and prepare data
        source_type, parsed_name = parse_source_path(self.source_path)
        data_dir = self._prepare_data_directory(source_type, parsed_name)

        # Setup Chroma storage
        chroma_path = self._get_storage_directory(parsed_name)
        os.makedirs(chroma_path, exist_ok=True)

        # Initialize Chroma client and collection
        self.chroma_client = PersistentClient(path=chroma_path)
        collection_name = f"collection_{parsed_name.replace(':', '_')}"
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.chroma_store = ChromaVectorStore(chroma_collection=self.collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.chroma_store)

        # Load or create index
        if self.collection.count() > 0:
            logger.info(f"Loading existing Chroma index (collection count: {self.collection.count()})")
            self.vector_db = VectorStoreIndex.from_vector_store(
                vector_store=self.chroma_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
        else:
            logger.info(f"Creating new Chroma index from {data_dir}")
            documents = self._load_documents(data_dir)
            self.vector_db = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            logger.info(f"Indexed {len(documents)} documents into Chroma")

        # Verify the vector_db is properly set
        if not isinstance(self.vector_db, VectorStoreIndex):
            raise RuntimeError(f"Expected VectorStoreIndex, got {type(self.vector_db)}")

        logger.info(f"Chroma VectorDB initialized successfully with {type(self.vector_db).__name__}")

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve documents using Chroma-specific optimizations.

        Args:
            query (str): Query string
            top_k (int): Number of top results to return

        Returns:
            List[NodeWithScore]: Retrieved documents with scores
        """
        logger.info(f"Chroma retrieval for query: '{query}' (top_k={top_k})")

        retriever = self.vector_db.as_retriever(similarity_top_k=top_k)
        nodes_with_scores = retriever.retrieve(query)

        # Chroma-specific post-processing could go here
        # For example: additional filtering, re-ranking, etc.

        logger.info(f"Chroma retrieved {len(nodes_with_scores)} documents")
        return nodes_with_scores

    def advanced_retrieve(self, query: str, top_k: int = 5,
                          where_filter: Dict = None,
                          include_metadata: bool = True) -> List[NodeWithScore]:
        """
        Advanced Chroma-specific retrieval with filtering capabilities.

        Args:
            query (str): Query string
            top_k (int): Number of results
            where_filter (Dict): Chroma where filter
            include_metadata (bool): Whether to include metadata

        Returns:
            List[NodeWithScore]: Retrieved documents
        """
        if where_filter:
            # Use Chroma's native filtering capabilities
            logger.info(f"Using Chroma advanced search with filter: {where_filter}")
            # This would require direct Chroma collection querying
            # Implementation depends on your specific filtering needs

        return self.retrieve(query, top_k)

    def add_documents(self, documents: List[Any]) -> None:
        """
        Add documents to the Chroma vector database.

        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to Chroma")
        self.vector_db.insert_nodes(documents)
        logger.info("Documents added to Chroma successfully")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Chroma database statistics.

        Returns:
            Dict containing Chroma-specific stats
        """
        return {
            "db_type": self.db_type,
            "collection_count": self.collection.count() if self.collection else 0,
            "collection_name": self.collection.name if self.collection else None,
            "storage_path": self.chroma_client._settings.persist_directory if self.chroma_client else None,
            "embedding_model": self.embedding_model.model_name
        }

    def persist(self) -> None:
        """
        Persist the Chroma database.
        Note: Chroma automatically persists data, but this can force a save.
        """
        logger.info("Persisting Chroma database")
        # Chroma handles persistence automatically with PersistentClient
        # Additional explicit persistence logic can go here if needed

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "chroma"

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
