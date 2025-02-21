from llama_index.core import GPTVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from config import DATA_PATH
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_index():
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    # Create a local embedding model using HuggingFace (SentenceTransformer)
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    # Build the vector store index using the local embedding model
    created_index = GPTVectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return created_index


# Testing function.
if __name__ == "__main__":
    index = create_index()
    print(">Index created with", len(index.docstore.docs), "documents.")
