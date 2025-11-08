import os
from typing import List, Dict, Any, Union

import numpy as np
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utility.distance_metrics import DistanceMetric
from utility.logger import logger
from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url, \
    load_local_dataset
from vector_db.vector_db_interface import VectorDBInterface

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SimpleVectorDB(VectorDBInterface):
    """
    Simple vector database implementation using LlamaIndex's default storage.
    """

    def __init__(self, source_path: str, embedding_model: HuggingFaceEmbedding, distance_metric: DistanceMetric):
        self.storage_dir = None
        super().__init__(source_path, embedding_model, distance_metric)
        if distance_metric != DistanceMetric.COSINE:
            logger.warning(f"[SimpleVectorDB] Only cosine similarity is supported. Ignoring '{distance_metric.value}'.")

    def _initialize(self) -> None:
        logger.info(f"Initializing Simple vector database for source: {self.source_path}")

        source_type, parsed_name = parse_source_path(self.source_path)

        self.storage_dir = self._get_storage_directory(parsed_name)
        os.makedirs(self.storage_dir, exist_ok=True)

        if self._index_exists():
            logger.info(f"Loading existing Simple vector database from {self.storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.vector_db = load_index_from_storage(storage_context, embed_model=self.embedding_model)
        else:
            data_dir = self._prepare_data_directory(source_type, parsed_name)
            logger.info(f"Creating new Simple vector database from {data_dir}")
            documents = load_local_dataset(parsed_name)
            self.vector_db = VectorStoreIndex.from_documents(documents, embed_model=self.embedding_model)
            self.persist()
            logger.info(f"Indexed {len(documents)} documents into Simple vector DB")

    def retrieve(self, query_or_vector: Union[str, np.ndarray], top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve documents using either a text query or an embedding vector.
        Note: Only cosine similarity is supported in this implementation.
        """
        if isinstance(query_or_vector, str):
            logger.info(f"Simple retrieval with text query: '{query_or_vector}' (top_k={top_k})")
            retriever = self.vector_db.as_retriever(similarity_top_k=top_k)
            return retriever.retrieve(query_or_vector)

        elif isinstance(query_or_vector, np.ndarray):
            logger.info("Simple retrieval with embedding vector")

            index_struct = self.vector_db.index_struct
            docstore = self.vector_db.docstore
            all_nodes = list(docstore.docs.values())

            if not all_nodes:
                logger.warning("No nodes available in Simple vector DB")
                return []

            embedded_nodes = self.vector_db._embedding_store._id_to_embedding
            all_embeddings = np.array([embedded_nodes[node.node_id] for node in all_nodes])

            if all_embeddings.shape[0] == 0:
                logger.error("No embeddings found in embedding store")
                return []

            norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_or_vector)
            similarities = np.dot(all_embeddings, query_or_vector) / (norms + 1e-10)
            top_indices = similarities.argsort()[::-1][:top_k]

            results = [
                NodeWithScore(node=all_nodes[i], score=similarities[i])
                for i in top_indices
            ]
            return results

        else:
            raise TypeError(f"Expected str or np.ndarray as query input, got {type(query_or_vector)}")

    def add_documents(self, documents: List[Any]) -> None:
        logger.info(f"Adding {len(documents)} documents to Simple vector DB")
        self.vector_db.insert_nodes(documents)
        self.persist()
        logger.info("Documents added to Simple vector DB successfully")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "db_type": self.db_type,
            "storage_dir": self.storage_dir,
            "embedding_model": self.embedding_model.model_name,
            "index_exists": self._index_exists()
        }

        try:
            if hasattr(self.vector_db, 'docstore') and hasattr(self.vector_db.docstore, 'docs'):
                stats["document_count"] = len(self.vector_db.docstore.docs)
        except Exception as e:
            logger.debug(f"Could not get document count: {e}")
            stats["document_count"] = "unknown"

        return stats

    def persist(self) -> None:
        logger.info(f"Persisting Simple vector database to {self.storage_dir}")
        self.vector_db.storage_context.persist(persist_dir=self.storage_dir)

    @property
    def db_type(self) -> str:
        return "simple"

    def _index_exists(self) -> bool:
        required_files = ["docstore.json", "index_store.json"]
        return (
                os.path.exists(self.storage_dir) and
                all(os.path.exists(os.path.join(self.storage_dir, f)) for f in required_files)
        )

    def _prepare_data_directory(self, source_type: str, parsed_name: str) -> str:
        if source_type == "url":
            data_dir = os.path.join(PROJECT_PATH, "data", "url_downloads", parsed_name)
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                logger.info(f"Downloading data from URL into {data_dir}")
                download_and_save_from_url(self.source_path, data_dir)
        elif source_type == "hf":
            data_dir = os.path.join(PROJECT_PATH, "data", "hf_downloads", parsed_name.replace(":", "_"))
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                logger.info(f"Downloading data from Hugging Face into {data_dir}")
                if ":" in self.source_path:
                    dataset_name, config = self.source_path.split(":", 1)
                else:
                    dataset_name, config = self.source_path, None
                download_and_save_from_hf(dataset_name, config, data_dir)
        else:
            data_dir = self.source_path

        return data_dir

#     def _load_documents(self, data_dir: str):
#         if not os.path.exists(data_dir) or not os.listdir(data_dir):
#             logger.error(f"No documents found in {data_dir}")
#             raise FileNotFoundError(f"No documents found in {data_dir}")
#
#         documents = SimpleDirectoryReader(data_dir).load_data()
#         if not documents:
#             logger.warning(f"No documents loaded from {data_dir}")
#             return []
#
#         return documents
