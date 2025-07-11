# modules/vector_db/chroma_index.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

PROJECT_PATH = os.path.abspath(__file__)


def build_chroma_vector_db(source: str, source_path: str, embedding_model) -> VectorStoreIndex:
    """
    Build a Chroma-based vector database vector_db.

    Args:
        source (str): 'local' or 'url'
        source_path (str): Path or identifier for the source.
        embedding_model: Embedding model to use.

    Returns:
        VectorStoreIndex: Index based on Chroma vector store.
    """
    if source == "url":
        corpus_name = source_path.split("/")[-1] if "://" in source_path else source_path.replace(":", "_")
        data_dir = os.path.join(PROJECT_PATH, "data", corpus_name)
        chroma_path = os.path.join(PROJECT_PATH, "storage", "chroma", corpus_name)
    else:
        data_dir = os.path.join(PROJECT_PATH, "data", "chroma")
        chroma_path = os.path.join(PROJECT_PATH, "storage", "chroma", "local")

    os.makedirs(chroma_path, exist_ok=True)

    client = PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="chroma_collection")
    chroma_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=chroma_store)

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(
            vector_store=chroma_store,
            embed_model=embedding_model,
            storage_context=storage_context
        )
    else:
        documents = SimpleDirectoryReader(data_dir).load_data()
        return VectorStoreIndex.from_documents(
            documents,
            embed_model=embedding_model,
            storage_context=storage_context
        )
