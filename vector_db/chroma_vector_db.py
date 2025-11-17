"""
Chroma Vector Database Implementation
"""

import os
import time
from typing import List, Dict, Any, Union

import numpy as np
from chromadb import PersistentClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from utility.distance_metrics import DistanceMetric
from utility.logger import logger
from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url, \
    load_local_dataset
from vector_db.vector_db_interface import VectorDBInterface

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

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding, distance_metric: DistanceMetric):

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
        self.storage_dir = None
        super().__init__(source_path, embedding_model, distance_metric)

    def _initialize(self) -> None:
        """Initialize the Chroma vector database."""
        logger.info(f"Initializing Chroma vector database for source: {self.source_path}")

        # Parse source and prepare data
        source_type, parsed_name = parse_source_path(self.source_path)

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
            metadata={
                "hnsw:space": self.distance_metric,
                "hnsw:M": 64,
                "hnsw:construction_ef": 200
            }
        )
        stored_metric = self.collection.metadata.get("hnsw:space")
        assert stored_metric == self.distance_metric, (
            f"❌ Mismatch between stored metric ({stored_metric}) and requested ({self.distance_metric})"
        )
        self.chroma_store = ChromaVectorStore(chroma_collection=self.collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.chroma_store)

        # Load or create index
        collection_count = self.collection.count()
        if collection_count > 0:
            logger.info(f"Existing documents in collection: {collection_count}. Loading existing index...")
            self.vector_db = VectorStoreIndex.from_vector_store(
                vector_store=self.chroma_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
        else:
            import json
            documents = load_local_dataset(parsed_name)
            logger.info(f"Starting indexing the documents into Chroma")
            total_docs = len(documents)
            batch_size = 10000
            added_count = 0
            skipped_count = 0
            start_time = time.time()
            self.vector_db = VectorStoreIndex.from_vector_store(
                vector_store=self.chroma_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )  # initialize empty index

            total_batches = (total_docs + batch_size - 1) // batch_size

            # --- Checkpointing additions ---
            checkpoint_path = os.path.join(chroma_path, "index_progress.json")
            last_indexed = 0
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r") as f:
                        checkpoint = json.load(f)
                        last_indexed = int(checkpoint.get("last_indexed", 0))
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
                    last_indexed = 0
            if last_indexed:
                print(f"🧩 Resuming from checkpoint: {last_indexed}/{total_docs}")
            else:
                print("🆕 Starting fresh indexing...")
            # --- End checkpointing additions ---

            # Only index batches that have not been done yet
            logger.info("⚙️ Running in sequential indexing mode (safe for MPS)")
            start_batch = last_indexed // batch_size
            pbar = tqdm(total=total_batches, desc="Indexing documents")
            # Advance progress bar to already indexed batches
            if start_batch > 0:
                pbar.update(start_batch)
            for batch_idx in range(start_batch, total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_docs)
                batch_docs = documents[start_idx:end_idx]
                try:
                    self.vector_db.insert_nodes(batch_docs)
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                batch_size_actual = end_idx - start_idx
                added_count += batch_size_actual
                elapsed = time.time() - start_time
                percent_done = (added_count / total_docs) * 100
                avg_time_per_doc = elapsed / added_count if added_count else 0
                remaining_docs = total_docs - added_count
                eta = remaining_docs * avg_time_per_doc

                # --- Checkpoint update ---
                import time as _time
                with open(checkpoint_path, "w") as f:
                    json.dump({"last_indexed": end_idx, "timestamp": _time.time()}, f)
                # --- End checkpoint update ---

                logger.info(
                    f"Progress: {percent_done:.2f}% | Batch {batch_idx + 1}/{total_batches} | ETA: {eta / 60:.1f} min")
                pbar.update(1)
            pbar.close()

            # After all indexing is done, remove checkpoint file
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint file: {e}")

            total_time = time.time() - start_time
            logger.info(f"Indexing complete: Added {added_count} documents, Skipped {skipped_count} documents, "
                        f"Total time: {total_time:.2f}s")

        # Verify the vector_db is properly set
        if not isinstance(self.vector_db, VectorStoreIndex):
            raise RuntimeError(f"Expected VectorStoreIndex, got {type(self.vector_db)}")

        logger.info(f"Chroma VectorDB initialized successfully with {type(self.vector_db).__name__}")

    def retrieve(self, query: Union[str, np.ndarray], top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve documents using direct Chroma collection query.

        Args:
            query (Union[str, np.ndarray]): Query text or embedding vector
            top_k (int): Number of top results to return

        Returns:
            List[NodeWithScore]: Retrieved documents with scores
        """
        logger.info(f"Retrieving top {top_k} documents from Chroma")
        logger.debug(f"Chroma direct query: {query} (top_k={top_k})")

        # Prepare query depending on type
        if isinstance(query, str):
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

        elif isinstance(query, np.ndarray):
            if query.ndim != 1:
                raise ValueError("Embedding query must be a 1D numpy array")

            vec = query.astype(np.float32)

            results = self.collection.query(
                query_embeddings=[vec.tolist()],
                n_results=top_k
            )
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        # No results case
        if not results["ids"] or not results["ids"][0]:
            logger.warning("No results returned from Chroma")
            return []

        # Convert Chroma results to NodeWithScore
        nodes_with_scores = []
        for doc_id, document, metadata, score in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
        ):
            node = TextNode(
                text=document,
                id_=doc_id,
                metadata=metadata
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        logger.info(f"Chroma returned {len(nodes_with_scores)} results")
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
