import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from modules.query import batch_generate_embeddings
from utility.logger import logger


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    sim = np.clip(cosine_similarity(a, b)[0][0], -1.0, 1.0)
    return float(1.0 - sim)


class TrilaterationRetriever:
    def __init__(
            self,
            embedding_model,
            vector_db,
            evaluator_model,
            top_k_candidates=25,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.evaluator_model = evaluator_model
        self.top_k_candidates = top_k_candidates

    def _compute_score_weighted_centroid(self, anchors_embs: np.ndarray, scores: np.ndarray):
        weights = softmax(scores)
        weighted_sum = np.sum(weights[:, None] * anchors_embs, axis=0)
        return weighted_sum

    def retrieve(self, query: str, query_emb: np.ndarray = None):
        """
        Retrieve best document using Trilateration (Weighted Centroid).

        Args:
            query: The text query.
            query_emb: (Optional) Pre-computed query embedding.
                       Pass this to save time if you already calculated it.
        """
        # ---------------------------------------------------------
        # OPTIMIZATION 1: Avoid Double Query Embedding
        # ---------------------------------------------------------
        if query_emb is None:
            query_emb = self.embedding_model.get_query_embedding(query)

        # 1. Standard Vector Retrieval
        nodes_with_scores = self.vector_db.retrieve(query_emb, top_k=self.top_k_candidates)

        if not nodes_with_scores:
            return None

        nodes = [n.node for n in nodes_with_scores]

        # 2. Prepare Text for Evaluator
        texts = []
        for n in nodes:
            content = getattr(n, "get_content", lambda: getattr(n, "text", ""))()
            texts.append(content)

        # ---------------------------------------------------------
        # OPTIMIZATION 2: Use Pre-Computed Document Embeddings
        # ---------------------------------------------------------
        candidates_embs = []
        missing_emb_indices = []

        for i, node in enumerate(nodes):
            if hasattr(node, "embedding") and node.embedding is not None:
                # ✅ Fast path: Use DB vector
                candidates_embs.append(node.embedding)
            else:
                # ⚠️ Slow path: Mark for re-calculation
                candidates_embs.append(None)
                missing_emb_indices.append(i)

        # Fallback: Generate only missing embeddings
        if missing_emb_indices:
            logger.debug(f"⚠️ DB missing embeddings for {len(missing_emb_indices)} nodes. Re-calculating...")
            missing_texts = [texts[i] for i in missing_emb_indices]
            new_embs = batch_generate_embeddings(missing_texts, self.embedding_model)

            for idx, emb in zip(missing_emb_indices, new_embs):
                candidates_embs[idx] = emb

        candidates_embs = np.array(candidates_embs)

        # 3. Run Evaluator (Cross-Encoder)
        pairs = [[query, text] for text in texts]
        evaluator_scores = self.evaluator_model.predict(pairs)

        # 4. Compute Weighted Centroid
        x_star = self._compute_score_weighted_centroid(candidates_embs, evaluator_scores)

        # 5. Find the document closest to the new Centroid
        distances = [cosine_distance(x_star, emb) for emb in candidates_embs]

        best_idx = np.argmin(distances)
        best_doc = nodes[best_idx]

        return {
            "query": query,
            "intersection": x_star,
            "best_doc": best_doc,
            "evaluator_score": evaluator_scores[best_idx],
            "distance_to_centroid": distances[best_idx]
        }
