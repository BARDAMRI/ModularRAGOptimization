# indexer/query_optimizer.py

def optimize_query(query):
    """
    Optimize the given query.
    This is a placeholder function where you can implement a more advanced optimization algorithm.

    Args:
        query (str): The original query.

    Returns:
        str: The optimized query.
    """
    # For now, just return the query unchanged.
    return query


def search_index(query, index):
    """
    Search the index for documents matching the optimized query.
    This function currently performs a simple substring search.

    Args:
        query (str): The search query.
        index (dict): The index mapping file names to content.

    Returns:
        dict: A dictionary of matching documents.
    """
    optimized_query = optimize_query(query)
    results = {}
    for file_name, content in index.items():
        if optimized_query.lower() in content.lower():
            results[file_name] = content
    return results