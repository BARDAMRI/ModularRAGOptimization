# main.py
from indexer.file_indexer import FileIndexer
from indexer.query_optimizer import search_index
from utils.logger import log_info


def main():
    log_info("Starting LlamaIndex application.")

    # Build the index from the data directory
    indexer = FileIndexer()
    index = indexer.build_index()
    log_info("Index built successfully.")

    # Prompt user for a query and search the index
    query = input("Enter your query: ")
    results = search_index(query, index)

    if results:
        print("Results found in the following files:")
        for file_name in results.keys():
            print(f"- {file_name}")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()