import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from configurations.config import TRILATERATION_MODE, NUM_BASE_ANCHORS, LIN_SOLVER_EPSILON, DISTANCE_SCALE_GAMMA, \
    LIN_SOLVER_MAX_ITERATIONS, LIN_SOLVER_TOLERANCE, LIN_SOLVER_STEP_SIZE
from utility.logger import logger


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    sim = np.clip(cosine_similarity(a, b)[0][0], -1.0, 1.0)
    return float(1.0 - sim)


class TrilaterationRetriever:
    """
    Full Trilateration-Based Retriever implementing:
    (0) Base-anchor initialization
    (1–7) Iterative geometric guessing loop with LLM scoring
    """

    def __init__(
            self,
            embedding_model,
            vector_db,
            evaluator_model,
            top_k_candidates=50,
            max_iterations=15,
            score_threshold=0.92,
            no_improve_patience=3,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.evaluator_model = evaluator_model
        self.top_k_candidates = top_k_candidates
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.no_improve_patience = no_improve_patience

        # ---------------------------------------------------------
        #  STAGE 0 — CREATE BASE ANCHORS (precomputed once)
        # ---------------------------------------------------------
        logger.info("Initializing Trilateration Retriever — selecting base anchors (Stage 0)...")

        # Step 0.1 — pull a large random batch of documents
        initial_candidates = self.vector_db.retrieve(
            np.random.randn(self.embedding_model.embedding_dim).astype(np.float32),
            top_k=2000
        )

        nodes = [n.node for n in initial_candidates]
        embs = [n.node.embedding for n in initial_candidates if n.node.embedding is not None]
        embs = np.array(embs)

        # Step 0.2 — pick first anchor randomly
        base_anchors = []
        used_idx = set()

        seed_idx = np.random.randint(0, len(embs))
        base_anchors.append(nodes[seed_idx])
        used_idx.add(seed_idx)

        # Step 0.3 — farthest-first selection for NUM_BASE_ANCHORS
        while len(base_anchors) < NUM_BASE_ANCHORS:
            best_idx = None
            best_dist = -1.0

            for i in range(len(embs)):
                if i in used_idx:
                    continue

                d = min(cosine_distance(embs[i], a.embedding) for a in base_anchors)
                if d > best_dist:
                    best_dist = d
                    best_idx = i

            if best_idx is None:
                break

            base_anchors.append(nodes[best_idx])
            used_idx.add(best_idx)

        # metadata
        for a in base_anchors:
            a.metadata["role"] = "base_anchor"
            a.metadata["iteration"] = 0

        self.base_anchors = base_anchors
        logger.info(f"Base anchors created: {len(base_anchors)} total.")


    # ---------------------------------------------------------
    #  STAGE 1–7 — MAIN ITERATIVE RETRIEVAL
    # ---------------------------------------------------------
    def retrieve(self, query: str):
        """
        Full iterative trilateration-based retrieval.
        """

        # keep runtime anchors
        anchors = list(self.base_anchors)

        best_score = -1.0
        best_doc = None
        no_improve_counter = 0

        for iteration in range(self.max_iterations):

            # ---------------------------------------------------------
            #  STAGE 2 — Score all anchors by LLM evaluator
            # ---------------------------------------------------------
            scores = []
            for a in anchors:
                prompt = (
                    "Rate relevance of DOCUMENT to QUERY (0.0–1.0).\n"
                    f"QUERY:\n{query}\n\n"
                    f"DOCUMENT:\n{a.text}\n\n"
                    "Return only a float."
                )
                raw = self.evaluator_model(prompt)
                try:
                    score = float(str(raw).strip())
                except:
                    score = 0.0
                scores.append(score)
                a.metadata["score"] = score
                a.metadata["iteration"] = iteration

            scores_np = np.array(scores)

            # update best document
            top_idx = int(np.argmax(scores_np))
            if scores_np[top_idx] > best_score:
                best_score = scores_np[top_idx]
                best_doc = anchors[top_idx]
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # stopping condition 1: high enough score
            if best_score >= self.score_threshold:
                return {"best_doc": best_doc, "score": best_score}

            # stopping condition 2: no improvement
            if no_improve_counter >= self.no_improve_patience:
                return {"best_doc": best_doc, "score": best_score}

            # ---------------------------------------------------------
            #  STAGE 3 — Compute new reference point x_t
            # ---------------------------------------------------------
            emb_matrix = np.array([a.embedding for a in anchors])

            if TRILATERATION_MODE == "metric_least_squares":
                x_t = self._metric_least_squares_point(anchors, scores_np)
            else:
                weights = softmax(scores_np)
                x_t = np.sum(weights[:, None] * emb_matrix, axis=0)

            # ---------------------------------------------------------
            #  STAGE 4 — Choose next document = nearest to x_t
            # ---------------------------------------------------------
            next_candidates = self.vector_db.retrieve(x_t, top_k=1)
            if not next_candidates:
                break

            next_node = next_candidates[0].node
            next_node.metadata["role"] = "dynamic_anchor"

            # ---------------------------------------------------------
            #  STAGE 5 & 6 — Evaluate & add new anchor
            # ---------------------------------------------------------
            prompt = (
                "Rate relevance of DOCUMENT to QUERY (0.0–1.0).\n"
                f"QUERY:\n{query}\n\n"
                f"DOCUMENT:\n{next_node.text}\n\n"
                "Return only a float."
            )
            raw = self.evaluator_model(prompt)
            try:
                next_score = float(str(raw).strip())
            except:
                next_score = 0.0

            next_node.metadata["score"] = next_score
            next_node.metadata["iteration"] = iteration

            anchors.append(next_node)

        # fallback after max iterations
        return {"best_doc": best_doc, "score": best_score}


    # ---------------------------------------------------------
    #  METRIC LEAST SQUARES INTERSECTION (STAGE 3 OPTION B)
    # ---------------------------------------------------------
    def _metric_least_squares_point(self, anchors, scores_np):
        """
        Compute x_t via robust metric least-squares trilateration.
        """

        A = np.array([a.embedding for a in anchors])
        r = DISTANCE_SCALE_GAMMA * (1 - scores_np)
        w = 1.0 / (r + LIN_SOLVER_EPSILON)

        # warm start
        x = (w[:, None] * A).sum(axis=0) / w.sum()

        for _ in range(LIN_SOLVER_MAX_ITERATIONS):
            diffs = x[None, :] - A
            dist = np.linalg.norm(diffs, axis=1)
            delta = dist - r
            denom = dist + LIN_SOLVER_EPSILON
            factors = 2 * w * delta / denom
            grad = (factors[:, None] * diffs).sum(axis=0)

            if np.linalg.norm(grad) <= LIN_SOLVER_TOLERANCE:
                break

            x = x - LIN_SOLVER_STEP_SIZE * grad

        return x
