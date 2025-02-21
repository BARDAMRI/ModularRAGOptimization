
from llama_index.core import GPTVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from config import DATA_PATH


def create_index():
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index

