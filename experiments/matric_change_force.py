import os
import shutil
import numpy as np
from chromadb import PersistentClient

# === Configuration ===
cosine_dir = "tmp_chroma_cosine"
euclidean_dir = "tmp_chroma_euclidean"
collection_name = "demo"

texts = ["This is a cat", "This is a dog", "This is a frog", "This is a robot"]
ids = [f"id_{i}" for i in range(len(texts))]

# Generate normalized embeddings (unit vectors) to emphasize metric differences
vecs = np.random.randn(len(texts), 768).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
query_vec = vecs[0]  # We'll use the first vector as query

def create_collection_with_metric(path: str, metric: str):
    """Create a Chroma collection with a specific distance metric."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    client = PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": metric,
            "hnsw:M": 64,
            "hnsw:construction_ef": 100
        }
    )
    return collection

# === Build collections with different distance metrics ===
print("Creating a cosine-based vector database...")
col_cosine = create_collection_with_metric(cosine_dir, "cosine")
col_cosine.add(documents=texts, embeddings=vecs.tolist(), ids=ids)

print("Creating a euclidean-based vector database...")
col_euclidean = create_collection_with_metric(euclidean_dir, "l2")
col_euclidean.add(documents=texts, embeddings=vecs.tolist(), ids=ids)

# === Query both collections with the same vector ===
print("\nQuerying cosine-based collection...")
res_cosine = col_cosine.query(query_embeddings=[query_vec.tolist()], n_results=4)

print("Querying euclidean-based collection...")
res_euclidean = col_euclidean.query(query_embeddings=[query_vec.tolist()], n_results=4)

# === Print distances from both collections ===
print("\nCosine Results (same vector, cosine distance):")
for doc, score in zip(res_cosine["documents"][0], res_cosine["distances"][0]):
    print(f"{doc} — Distance: {score:.4f}")

print("\nEuclidean Results (same vector, euclidean distance):")
for doc, score in zip(res_euclidean["documents"][0], res_euclidean["distances"][0]):
    print(f"{doc} — Distance: {score:.4f}")

# === Try to forcefully change the distance metric after creation ===
print("\nAttempting to change the cosine collection's distance metric to 'l2' in memory...")
col_cosine.metadata["hnsw:space"] = "l2"

print("Querying again after changing metadata (should still behave as cosine)...")
res_after_change = col_cosine.query(query_embeddings=[query_vec.tolist()], n_results=4)

print("\nResults after attempting to change metric (still using cosine behavior):")
for doc, score in zip(res_after_change["documents"][0], res_after_change["distances"][0]):
    print(f"{doc} — Distance: {score:.4f}")

# === Print stored metadata to show it was not truly changed ===
print("\nStored metadata in cosine collection:")
print(col_cosine.metadata)
