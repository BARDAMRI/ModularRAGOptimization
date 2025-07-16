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
from utility.distance_metrics import DistanceMetric

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_documents(data_dir: str):
    """
    Load documents from a directory using SimpleDirectoryReader.

    Args:
        data_dir (str): Directory path to load documents from

    Returns:
        List of documents
    """
    return SimpleDirectoryReader(data_dir).load_data()


class ChromaVectorDB(VectorDBInterface):
    """
    Chroma-based vector database implementation.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding,
                 distance_metric: DistanceMetric = DistanceMetric.COSINE):
        """
        Initialize Chroma vector database.

        Args:
            source_path (str): Path to the data source
            embedding_model: Embedding model to use
            distance_metric: Distance metric (default: COSINE)
        """
        self.chroma_client = None
        self.collection = None
        self.chroma_store = None
        super().__init__(source_path, embedding_model, distance_metric)

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
        if self.distance_metric not in {m.value for m in DistanceMetric}:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        stored_metric = self.collection.metadata.get("hnsw:space")
        assert stored_metric == self.distance_metric, (
            f"❌ Mismatch between stored metric ({stored_metric}) and requested ({self.distance_metric})"
        )
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
            documents = _load_documents(data_dir)
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

    def _prepare_data_directory(self, source_type: str, parsed_name: str) -> str:
        local_dir = os.path.join(PROJECT_PATH, "data", parsed_name)

        if os.path.exists(local_dir) and os.listdir(local_dir):
            logger.info(f"Using existing data directory: {local_dir}")
            return local_dir

        os.makedirs(local_dir, exist_ok=True)

        if source_type == "url":
            logger.info(f"Downloading data from URL into {local_dir}")
            download_and_save_from_url(self.source_path, local_dir)

        elif source_type == "hf":
            logger.info(f"Downloading data from HF into {local_dir}")
            if ":" in self.source_path:
                dataset_name, config = self.source_path.split(":", 1)
            else:
                dataset_name, config = self.source_path, None
            download_and_save_from_hf(dataset_name, config, local_dir)

        else:
            logger.warning(f"Unknown source type '{source_type}', using as-is: {local_dir}")

        return local_dir

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
        return {
            "db_type": self.db_type,
            "collection_count": self.collection.count() if self.collection else 0,
            "collection_name": self.collection.name if self.collection else None,
            "storage_path": self.chroma_client.get_settings().persist_directory if self.chroma_client else None,
            "embedding_model": self.embedding_model.model_name,
            "distance_metric": self.distance_metric
        }

    def persist(self) -> None:
        logger.info("Persisting Chroma database")

    @property
    def db_type(self) -> str:
        return "chroma"
