import os
import shutil
import tempfile
import pytest
from chromadb import PersistentClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vector_db.chroma_vector_db import ChromaVectorDB
from llama_index.core.schema import TextNode


@pytest.fixture
def temp_chroma_dir():
    # Temporary directory for Chroma storage
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def embedding_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_chroma_allows_euclidean_distance(temp_chroma_dir, embedding_model):
    # Create dummy documents
    documents = [TextNode(text="apple"), TextNode(text="banana"), TextNode(text="fruit salad")]

    # Create a fake local source directory with dummy files
    fake_data_dir = os.path.join(temp_chroma_dir, "dummy_data")
    os.makedirs(fake_data_dir, exist_ok=True)
    with open(os.path.join(fake_data_dir, "doc.txt"), "w") as f:
        f.write("This is a test document about fruit.")

    # Initialize the DB with Euclidean distance
    vector_db = ChromaVectorDB(
        source_path=fake_data_dir,
        embedding_model=embedding_model,
        distance_metric="l2"  # Euclidean distance in Chroma
    )
    vector_db._initialize()  # Explicitly call to force index creation

    # Add documents
    vector_db.add_documents(documents)

    # Test retrieval
    results = vector_db.retrieve("apple", top_k=2)
    assert len(results) > 0
    assert all(
        "apple" in str(node.get_content()).lower() or "fruit" in str(node.get_content()).lower() for node in results)

    # Check Chroma collection metadata
    client = PersistentClient(path=vector_db.get_stats()["storage_path"])
    collection = client.get_collection(name=vector_db.collection.name)
    assert collection.name.startswith("collection_")
