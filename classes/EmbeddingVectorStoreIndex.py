from llama_index.core import VectorStoreIndex
from typing import List, Tuple, Dict, Callable
import numpy as np


class EmbeddingVectorStoreIndex:
    def __init__(self, index: VectorStoreIndex, embed_fn: Callable[[str], np.ndarray]):
        """
        A wrapper around VectorStoreIndex that allows access to document embeddings.

        Args:
            index (VectorStoreIndex): The original LlamaIndex vector index.
            embed_fn (Callable[[str], np.ndarray]): Function that computes the embedding from text.
        """
        self.index = index
        self.embed_fn = embed_fn
        self.embeddings: Dict[str, np.ndarray] = {}

        # Extract and store embeddings from all documents in the docstore
        all_node_ids = list(index.docstore.docs.keys())
        for node in index.docstore.get_nodes(all_node_ids):
            if node.text:
                try:
                    embedding = embed_fn(node.text)
                    self.embeddings[node.node_id] = np.array(embedding)
                except Exception as e:
                    print(f"Failed to embed node {node.node_id[:6]}: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, np.ndarray]]:
        """
        Retrieve top-k nodes using the vector index, along with their scores and precomputed embeddings.

        Returns:
            List of (text, score, embedding) tuples.
        """
        nodes = self.index.as_retriever(similarity_top_k=top_k).retrieve(query)
        return [
            (node.node.text, node.score, self.embeddings.get(node.node.node_id))
            for node in nodes if node.node.node_id in self.embeddings
        ]

    def get_embedding(self, node_id: str) -> np.ndarray:
        """
        Get the embedding for a specific node/document ID.
        """
        return self.embeddings[node_id]
