import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utility.vector_db_utils import parse_source_path, download_and_save_from_hf, download_and_save_from_url

PROJECT_PATH = os.path.abspath(__file__)


def build_simple_vector_db(source: str, source_path: str, embedding_model: HuggingFaceEmbedding,
                           logger) -> BaseIndex | VectorStoreIndex:
    """
    Build a simple vector database index using GPTVectorStoreIndex with persistent storage.

    Args:
        source (str): 'local' or 'url'
        source_path (str): Path or identifier for the source
        embedding_model: Embedding model to use

    Returns:
        VectorStoreIndex: The loaded or newly created vector store index
    """
    if source == "url":
        if source_path is None:
            logger.error("source_path must be provided for 'url' source.")
            raise ValueError("source_path must be provided for 'url' source.")

        source_type, corpus_name = parse_source_path(source_path)
        corpus_dir = os.path.join(PROJECT_PATH, "data", corpus_name)
        storage_dir = os.path.join(PROJECT_PATH, "storage", "simple", corpus_name)

        if not os.path.exists(corpus_dir):
            logger.info(f"Downloading corpus into {corpus_dir}...")
            if source_type == "hf":
                dataset, config = source_path.split(":", 1)
                download_and_save_from_hf(dataset, config, corpus_dir)
            else:
                download_and_save_from_url(source_path, corpus_dir)
    else:
        corpus_dir = os.path.join(PROJECT_PATH, "data", "simple")
        storage_dir = os.path.join(PROJECT_PATH, "storage", "simple")

    os.makedirs(storage_dir, exist_ok=True)

    # Try to load existing index
    if os.path.exists(storage_dir) and os.path.exists(os.path.join(storage_dir, "docstore.json")):
        logger.info(f"Loading existing simple vector database from {storage_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context, embed_model=embedding_model)

    # Create new index
    logger.info(f"Indexing documents from {corpus_dir}...")
    documents = SimpleDirectoryReader(corpus_dir).load_data()
    vector_db = VectorStoreIndex.from_documents(
        documents,
        embed_model=embedding_model
    )

    # Persist the index
    vector_db.storage_context.persist(persist_dir=storage_dir)
    logger.info(f"Indexed {len(documents)} documents and saved to {storage_dir}.")

    return vector_db
