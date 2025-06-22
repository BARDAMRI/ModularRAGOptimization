import inspect
from modules.indexer import load_vector_db
from config import INDEX_SOURCE_URL

# Load vector DB and embedding model
vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

# Print constructor of the vector DB
print("=== Vector DB Class Source ===")
print(inspect.getsource(vector_db.__class__))

# Print embedding vectorization method (usually `encode`)
if hasattr(embedding_model, 'encode'):
    print("\n=== Embedding Model encode() Source ===")
    print(inspect.getsource(embedding_model.encode))
else:
    print("❗Embedding model does not have an 'encode' method.")

# Also check __call__ if encode is missing or just to be sure
if hasattr(embedding_model, '__call__'):
    print("\n=== Embedding Model __call__() Source ===")
    print(inspect.getsource(embedding_model.__call__))