import numpy as np
from numpy.linalg import norm
import inspect

from config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from modules.query import vector_similarity
from utility.embedding_utils import get_query_vector
from llama_index.core.vector_stores.simple import SimpleVectorStore


def normalize(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v


# Load the vector database and embedding model
vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

# # Access internal vector store
vector_store = getattr(vector_db, "_vector_store", None)
# print(f"Vector store type: {type(vector_store)}")
# print(inspect.getsource(SimpleVectorStore.query))  # Optional: Show source for verification

# Prepare a query
query = "This is a sample query."

# Convert query into vector using the same embedding model used in the index
query_vector = get_query_vector(query, embedding_model)
query_vector = normalize(query_vector)

# Query the VectorStoreIndex for the closest document
nodes = vector_db.as_retriever(similarity_top_k=1).retrieve(query)
node_with_score = nodes[0]
retrieved_node = node_with_score.node
retrieved_score = node_with_score.score

# Get the stored vector used in the index for this document
stored_embedding = vector_store.data.embedding_dict[retrieved_node.node_id]
stored_embedding = normalize(np.array(stored_embedding))

# Compute cosine similarity manually
manual_similarity_score = np.dot(query_vector, stored_embedding)

# Print comparison
print(f"VectorStoreIndex score: {retrieved_score}")
print(f"Manually computed cosine: {manual_similarity_score}")
print(f"Difference: {abs(retrieved_score - manual_similarity_score)}")
