"""
Chroma Vector Database Implementation
"""

import os
from typing import List, Dict, Any
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url
from utility.logger import logger
from vector_db.vector_db_interface import VectorDBInterface
from utility.distance_metrics import DistanceMetric
import logging

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_documents(data_dir: str):
    """
    Load documents from a directory using SimpleDirectoryReader.

    Args:
        data_dir (str): Directory path to load documents from

    Returns:
        List of documents
    """
    # SimpleDirectoryReader handles basic document loading, but not advanced chunking
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
        logger = logging.getLogger(__name__)  # Use the logger for this class instance

    def _initialize(self) -> None:
        """Initialize the Chroma vector database."""
        logger.info(f"Initializing Chroma vector database for source: {self.source_path}")

        # Parse source and prepare data
        source_type, parsed_name = parse_source_path(self.source_path)
        data_dir = self._prepare_data_directory(source_type, parsed_name)

        # Setup Chroma storage path
        chroma_path = os.path.join(PROJECT_PATH, "chroma_db_data", parsed_name.replace(':', '_'))
        os.makedirs(chroma_path, exist_ok=True)

        # Initialize Chroma client
        # Using Settings to explicitly set persist_directory and disable telemetry for consistency
        self.chroma_client = PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
        collection_name = f"collection_{parsed_name.replace(':', '_')}"

        # Validate distance metric
        if self.distance_metric not in {m for m in DistanceMetric}:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric.value}")

        # HNSW configuration parameters for collection creation
        # These affect the quality of the graph built during indexing
        hnsw_config = {
            "hnsw:space": self.distance_metric.value,
            "hnsw:M": 128,  # Max connections per node
            "hnsw:construction_ef": 128,  # Effectiveness factor during index construction
        }
        logger.info(f"HNSW indexing configuration: {hnsw_config}")

        # --- CRITICAL CHANGE FOR DETERMINISM HERE ---
        # 1. Attempt to delete the collection if it exists, to ensure a clean slate.
        # This is essential for experiments requiring full determinism and DB state reset.
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(
                f"Deleted existing Chroma collection: '{collection_name}' to ensure a clean state for this run.")
        except Exception as e:
            # If the collection does not exist, simply proceed
            logger.info(
                f"Collection '{collection_name}' did not exist or could not be deleted ({e}). Proceeding to create.")
        # --- END CRITICAL CHANGE ---

        logger.info(f"Attempting to get or create collection '{collection_name}'.")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_model,  # LlamaIndex uses this directly
            metadata=hnsw_config
        )

        # Basic check for metric consistency if collection already existed (should not happen after delete)
        stored_metric = self.collection.metadata.get("hnsw:space")
        if stored_metric != self.distance_metric.value:
            raise RuntimeError(
                f"❌ Mismatch between stored metric ({stored_metric}) and requested ({self.distance_metric.value}). "
                f"This indicates an issue with collection deletion/creation or configuration."
            )
        self.chroma_store = ChromaVectorStore(chroma_collection=self.collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.chroma_store)

        # Always build a new index if the collection was deleted or is empty
        if self.collection.count() == 0:
            logger.info(f"Collection is empty. Creating new Chroma index from {data_dir}")
            documents = _load_documents(data_dir)
            self.vector_db = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            logger.info(
                f"Indexed {len(documents)} documents into Chroma. New collection count: {self.collection.count()}")
        else:
            # This branch should ideally not be hit if delete_collection worked.
            # Log a warning if it does, as it implies a non-clean state.
            logger.warning(
                f"Collection '{collection_name}' unexpectedly has {self.collection.count()} documents "
                f"after intended deletion. Loading existing index, which might lead to inconsistencies if not intended."
            )
            self.vector_db = VectorStoreIndex.from_vector_store(
                vector_store=self.chroma_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )

        # Ensure the client persists changes to disk immediately
        self.chroma_client.persist()

        # Verify the vector_db is properly set
        if not isinstance(self.vector_db, VectorStoreIndex):
            raise RuntimeError(f"Expected VectorStoreIndex, got {type(self.vector_db).__name__}")

        logger.info(f"Chroma VectorDB initialized successfully with {type(self.vector_db).__name__}")

    def add(self, embeddings: List[List[float]], metadatas: List[dict], ids: List[str]):
        """
        Adds embeddings, metadatas, and ids to the ChromaDB collection.
        This method is primarily for direct additions to the underlying collection.
        For adding LlamaIndex Documents/Nodes, use add_documents.
        """
        logger.info(f"ChromaVectorDB.add: Attempting to add {len(ids)} raw embeddings.")
        if len(ids) > 0:
            logger.debug(
                f"First ID: {ids[0]}, First Embedding shape: {np.array(embeddings[0]).shape}, First Metadata: {metadatas[0]}")

        processed_metadatas = []
        for i, meta in enumerate(metadatas):
            # ChromaDB's .add method handles 'ids' list directly.
            # Including 'id' in metadata as well can be useful for LlamaIndex Node consistency.
            if "id" not in meta:
                meta["id"] = ids[i]
            processed_metadatas.append(meta)

        self.collection.add(
            embeddings=embeddings,
            metadatas=processed_metadatas,
            ids=ids
        )
        logger.info(
            f"Added {len(ids)} documents directly to ChromaDB collection '{self.collection.name}'. Current count: {self.collection.count()}")
        self.chroma_client.persist()  # Persist changes after adding

    def retrieve(self, query_or_vector: Any, top_k: int = 5) -> List[NodeWithScore]:
        """
        Unified retrieve function: handles both string queries and embedding vectors.
        Utilizes ChromaDB's ANN capabilities for efficient retrieval.

        Args:
            query_or_vector (str or np.ndarray): Text query or embedding vector
            top_k (int): Number of top results to return

        Returns:
            List[NodeWithScore]: Retrieved documents with scores
        """
        logger.info(f"ChromaVectorDB.retrieve called with top_k={top_k}")

        # HNSW search effectiveness parameter
        # Increased 'ef' to improve determinism and accuracy of search results.
        # Experiment with values: 512, 1024, 2048 - higher values mean more accurate (and slower) search.
        search_ef_value = 512
        logger.debug(f"Using HNSW search_ef: {search_ef_value}")

        results = None
        if isinstance(query_or_vector, str):
            logger.debug(f"Chroma retrieval using query_texts (ANN) for: '{query_or_vector}'")
            results = self.collection.query(
                query_texts=[query_or_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                ef=search_ef_value
            )

        elif isinstance(query_or_vector, np.ndarray):
            logger.debug("Chroma retrieval using query_embeddings (ANN)")
            # Ensure the vector is normalized if your distance metric expects it (e.g., Cosine)
            # and that the embedding model also normalizes outputs.
            norm = np.linalg.norm(query_or_vector)
            normalized_query_vector = query_or_vector / norm if norm != 0 else query_or_vector

            logger.debug(f"Query embedding norm: {norm:.4f}")
            if np.isnan(normalized_query_vector).any() or np.isinf(normalized_query_vector).any():
                logger.warning("Query vector contains NaN or Inf values after normalization! Using as-is.")

            results = self.collection.query(
                query_embeddings=[normalized_query_vector.tolist()],  # Convert numpy array to list
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                ef=search_ef_value
            )

        else:
            raise TypeError("Input must be either a string or a NumPy array for Chroma retrieval.")

        nodes: List[NodeWithScore] = []
        if results and results["documents"] and results["documents"][0]:
            logger.debug(f"Chroma raw query results length: {len(results['documents'][0])}")
            for doc, metadata, distance in zip(results["documents"][0], results["metadatas"][0],
                                               results["distances"][0]):
                # Determine score based on distance metric
                score = 0.0
                if self.distance_metric == DistanceMetric.COSINE:
                    # Chroma's cosine distance is usually 1 - similarity.
                    # So, similarity = 1 - distance. Higher score for higher similarity.
                    score = 1.0 - distance
                elif self.distance_metric == DistanceMetric.EUCLIDEAN:
                    # For Euclidean (L2), a common conversion to similarity is 1 / (1 + distance)
                    # Higher score for lower distance.
                    score = 1.0 / (1.0 + distance)
                elif self.distance_metric == DistanceMetric.INNER_PRODUCT:
                    # For Inner Product, Chroma often returns similarity directly for normalized embeddings.
                    # If embeddings are normalized, IP is cosine similarity. Higher score for higher IP.
                    score = distance
                else:
                    logger.warning(f"Unknown distance metric {self.distance_metric}. Using raw distance as score.")
                    score = distance

                # Ensure 'id' is available in metadata for NodeWithScore
                node_id = metadata.get("id") if metadata else None
                if node_id is None:
                    # Fallback if ID is missing (should not happen if `add` or `add_documents` works correctly)
                    node_id = f"unknown_id_{hash(doc)}"
                    logger.warning(
                        f"Node ID missing in metadata for doc: {doc[:50]}..., assigning placeholder: {node_id}")

                node = TextNode(text=doc, metadata=metadata, id_=node_id)
                nodes.append(NodeWithScore(node=node, score=score))
                logger.debug(f"Retrieved Node ID: {node.id_}, Score: {score:.4f}, Raw Distance: {distance:.4f}")
        else:
            logger.warning("No documents retrieved from ChromaDB for the given query.")

        logger.info(f"Chroma retrieval returned {len(nodes)} documents.")
        return nodes

    def delete(self):
        """Deletes the collection."""
        if self.collection:
            logger.info(f"Deleting ChromaDB collection '{self.collection.name}'.")
            self.chroma_client.delete_collection(name=self.collection.name)
            self.collection = None  # Reset collection reference
            self.chroma_client.persist()  # Ensure deletion is persisted on disk
        else:
            logger.info("No ChromaDB collection to delete.")

    def advanced_retrieve(self, query: str, top_k: int = 5,
                          where_filter: Dict = None,
                          include_metadata: bool = True) -> List[NodeWithScore]:
        """
        Advanced Chroma-specific retrieval with filtering capabilities.

        Args:
            query (str): Query string
            top_k (int): Number of results
            where_filter (Dict): Chroma where filter dictionary
            include_metadata (bool): Whether to include metadata

        Returns:
            List[NodeWithScore]: Retrieved documents
        """
        if where_filter:
            logger.info(
                f"Using Chroma advanced search with filter: {where_filter}. (Currently delegates to basic retrieve)")
            # TODO: Implement actual 'where' filtering logic here if needed,
            # by passing 'where' and potentially 'where_document' parameters to self.collection.query.
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
        Add documents to the Chroma vector database via LlamaIndex's VectorStoreIndex.
        """
        logger.info(f"Adding {len(documents)} documents to Chroma via VectorStoreIndex.insert_nodes.")
        # This calls the underlying vector store's add_nodes method.
        self.vector_db.insert_nodes(documents)
        logger.info(f"Documents added to Chroma successfully. Current collection count: {self.collection.count()}")
        self.chroma_client.persist()  # Persist changes after adding documents

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the ChromaDB instance."""
        return {
            "db_type": self.db_type,
            "collection_count": self.collection.count() if self.collection else 0,
            "collection_name": self.collection.name if self.collection else None,
            "storage_path": self.chroma_client.get_settings().persist_directory if self.chroma_client else None,
            "embedding_model": self.embedding_model.model_name,
            "distance_metric": self.distance_metric.value
        }

    def persist(self) -> None:
        """Persists the Chroma database. For PersistentClient, changes are usually saved automatically."""
        logger.info(
            "Chroma database uses PersistentClient, changes are automatically persisted by underlying operations.")
        # Explicitly call persist for good measure, though many operations trigger it.
        self.chroma_client.persist()

    @property
    def db_type(self) -> str:
        return "chroma"
