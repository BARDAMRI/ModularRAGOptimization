import numpy as np

from config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from utility.embedding_utils import get_query_vector


# Helper to embed node consistently (using same method as vector DB)
def embed_node_consistently(text: str, model, vector_db):
    # Use as_query=False to match the document encoding during index creation
    embedding = model.get_text_embedding(text)
    return normalize(np.array(embedding))


def normalize(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v


# Load the vector database and embedding model
vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

# Access internal vector store
vector_store = getattr(vector_db, "_vector_store", None)

# Prepare a query
query = "This is a sample query."
query_vector = normalize(get_query_vector(query, embedding_model))

# Query the VectorStoreIndex
nodes = vector_db.as_retriever(similarity_top_k=1).retrieve(query)
node_with_score = nodes[0]
retrieved_node = node_with_score.node
retrieved_score = node_with_score.score
node_id = retrieved_node.node_id

# === TEXT CHECKS ===
original_text = retrieved_node.text

print("🔍 TEXT ANALYSIS")
print("-" * 40)
print("Original Text:")
print(repr(original_text))

# Optionally re-load from the same source file (if available) — skipped here
# Check for leading/trailing whitespace
stripped_text = original_text.strip()
if original_text != stripped_text:
    print("⚠️  Text has leading/trailing whitespace.")

# Normalize to remove all extra spaces
normalized_text = " ".join(original_text.split())
if original_text != normalized_text:
    print("⚠️  Text has extra internal spacing or newlines.")

print("Text length:", len(original_text))

# === EMBEDDING CHECKS ===
print("\n🔍 EMBEDDING ANALYSIS")
print("-" * 40)

# Get stored vector from the index
stored_vector = normalize(np.array(vector_store.data.embedding_dict[node_id]))

#
# Re-encode using the same embedding path as the vector DB
reencoded_doc_vector = embed_node_consistently(original_text, embedding_model, vector_db=vector_db)
print(f"Re-encoded vector preview: {reencoded_doc_vector[:5]}")
cosine_1 = np.dot(query_vector, reencoded_doc_vector)
cosine_vs_stored = np.dot(query_vector, stored_vector)

# Try using get_query_embedding on the doc (just to test!)
query_style_doc_vector = normalize(get_query_vector(original_text, embedding_model))
cosine_query_style = np.dot(query_vector, query_style_doc_vector)

# Distance between vectors
reencode_diff = np.linalg.norm(reencoded_doc_vector - stored_vector)

print(f"Stored cosine (from vector_store): {retrieved_score}")
print(f"Manual cosine w/ stored vector:   {cosine_vs_stored}")
print(f"Manual cosine w/ re-encoded doc:  {cosine_1}")
print(f"Manual cosine w/ query-style doc: {cosine_query_style}")
print(f"⚠️  Vector difference (||stored - reencoded||): {reencode_diff:.6f}")

if reencode_diff > 1e-5:
    print("❗Stored vector and regenerated vector do NOT match exactly.")

print("Stored vector dim:", stored_vector.shape)
print("Re-encoded vector dim:", reencoded_doc_vector.shape)
