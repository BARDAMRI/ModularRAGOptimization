import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from modules.query import extract_node_text_for_embedding, batch_generate_embeddings, \
    generate_embedding_with_normalization


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(1 - cosine_similarity(a, b)[0][0])


class TrilaterationRetriever:
    def __init__(
            self,
            embedding_model,
            vector_db,
            top_k_candidates=1000,
            k_init=3,
            iterative: bool = False,
            max_refine_steps: int = 3,
            convergence_tol: float = 1e-4,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k_candidates = top_k_candidates
        self.k_init = k_init
        self.iterative = iterative
        self.max_refine_steps = max_refine_steps
        self.convergence_tol = convergence_tol
        self.anchors_cache = None
        self.similarity_func = self.get_similarity_func(embedding_model)

    def get_similarity_func(self, embedding_model):
        if hasattr(embedding_model, "distance"):
            return embedding_model.distance
        elif hasattr(embedding_model, "similarity"):
            return embedding_model.similarity
        else:
            return cosine_distance

    def _has_converged(self, x_prev: np.ndarray, x_curr: np.ndarray, tol: float) -> bool:
        return np.linalg.norm(x_curr - x_prev) <= tol

    def _already_anchor(self, emb: np.ndarray, anchors: np.ndarray, eps: float = 1e-6) -> bool:
        if anchors.shape[0] == 0:
            return False
        return np.any(np.linalg.norm(anchors - emb, axis=1) <= eps)

    def encode(self, text: str):
        return self.embedding_model.encode(text)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance using embedding model's native method if available."""
        return self.similarity_func(a, b)

    def _select_diverse_anchors(self, query_emb, candidates_embs):
        anchors = [candidates_embs[0]]
        while len(anchors) < self.k_init:
            min_distances = []
            for emb in candidates_embs:
                distances_to_anchors = [self._distance(emb, anchor) for anchor in anchors]
                min_distances.append(min(distances_to_anchors))
            max_min_dist_idx = np.argmax(min_distances)
            anchors.append(candidates_embs[max_min_dist_idx])
        return np.array(anchors)

    def _compute_geometric_intersection(self, query_emb, anchors):
        """Compute weighted centroid (x_star) as geometric intersection."""
        r_a = np.array([self._distance(query_emb, anchor) for anchor in anchors])
        w_a = 1 / (r_a + 1e-8)
        weighted_sum = np.sum(w_a[:, None] * anchors, axis=0)
        x_star = weighted_sum / np.sum(w_a)
        return x_star

    def retrieve(self, query: str):
        query_emb = generate_embedding_with_normalization(query, self.embedding_model)
        nodes_with_scores = self.vector_db.retrieve(query_emb, top_k=self.top_k_candidates)
        nodes = [n.node for n in nodes_with_scores]
        texts = extract_node_text_for_embedding(nodes)
        candidates_embs = batch_generate_embeddings(texts, self.embedding_model)

        # Prepare anchors (from cache or fresh)
        if self.anchors_cache is None:
            anchors = self._select_diverse_anchors(query_emb, candidates_embs)
            if not self.iterative:
                self.anchors_cache = anchors
        else:
            anchors = self.anchors_cache

        if not self.iterative:
            # Non-iterative mode: standard behavior, do not mutate cache
            x_star = self._compute_geometric_intersection(query_emb, anchors)

        else:
            # Iterative refinement mode
            anchors = anchors.copy()
            for step in range(self.max_refine_steps):
                x_star = self._compute_geometric_intersection(query_emb, anchors)

                # Find closest candidate to x_star
                distances = [self._distance(x_star, emb) for emb in candidates_embs]
                best_idx = np.argmin(distances)
                best_emb = candidates_embs[best_idx]

                # Check if best_emb is already an anchor
                if self._already_anchor(best_emb, anchors):
                    break
                # Save previous intersection before updating anchors
                x_prev = x_star.copy()
                # Add best_emb to anchors
                anchors = np.vstack([anchors, best_emb])
                self.anchors_cache = anchors
                # Recompute intersection with new anchors in next iteration
                x_star_new = self._compute_geometric_intersection(query_emb, anchors)
                # Check convergence
                if self._has_converged(x_prev, x_star_new, self.convergence_tol):
                    x_star = x_star_new
                    break
                x_star = x_star_new

        # After loop (or non-iterative), select final best doc
        distances = [self._distance(x_star, emb) for emb in candidates_embs]
        best_idx = np.argmin(distances)
        best_doc = nodes[best_idx]
        best_distance = distances[best_idx]

        return {
            "query": query,
            "intersection": x_star,
            "best_doc": best_doc,
            "distance": best_distance
        }
