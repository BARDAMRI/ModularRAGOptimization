import csv

import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import CrossEncoder

from utility.logger import logger
from vector_db.trilateration_retriever import TrilaterationRetriever
from vector_db.vector_db_interface import VectorDBInterface


def run_retrieval_base_experiment(
        queries: list[dict],
        vector_db: VectorDBInterface,
        embed_model: BaseEmbedding,
        top_k: int,
        output_path: str,
        evaluator_model: CrossEncoder
) -> str:
    logger.info(f"🚀 Starting Retrieval Experiment with {len(queries)} queries...")

    # Initialize retriever
    tri = TrilaterationRetriever(
        embedding_model=embed_model,
        vector_db=vector_db,
        evaluator_model=evaluator_model,
        top_k_candidates=top_k
    )

    # Track aggregate metrics
    naive_hits = 0
    tri_hits = 0
    total = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "question", "ground_truth_id",
            "naive_id", "tri_id",
            "naive_hit", "tri_hit",  # <-- These are your REAL metrics
            "naive_conf_score", "tri_conf_score"
        ])

        for idx, entry in enumerate(queries):
            question = entry.get("question")
            gt_id = str(entry.get("pubid"))  # Normalize ID to string

            # =====================================================
            # FIX 1: Use Standard API for Embeddings
            # Do NOT use custom 'generate_embedding_with_normalization'
            # Use the model's native method to ensure space consistency
            # =====================================================
            q_emb = (embed_model.get_query_embedding(question))

            # 2. Create a Numpy version for your Trilateration Math
            if isinstance(q_emb_list, np.ndarray):
                q_emb_array = q_emb_list
                q_emb_list = q_emb_list.tolist()  # Ensure we have a list for DB
            else:
                q_emb_array = np.array(q_emb_list)
            # ---- 1. Naive Baseline ----
            naive_results = vector_db.retrieve(query=q_emb_array, top_k=1)

            naive_id = "MISSING"
            naive_score = 0.0

            if naive_results:
                node = naive_results[0].node
                naive_id = str(node.metadata.get("PMID") or node.metadata.get("pubid") or "UNKNOWN")

                # Get text for the evaluator (just for confidence checking)
                content = node.get_content()
                naive_score = float(evaluator_model.predict([[question, content]])[0])

            # ---- 2. Trilateration Retriever ----
            # The retriever internal logic must also be updated to use
            # embed_model.get_query_embedding(query) internally!
            tri_result = tri.retrieve(question, query_emb=q_emb_array)

            tri_id = "MISSING"
            tri_score = 0.0

            if tri_result and tri_result.get("best_doc"):
                node = tri_result["best_doc"]
                tri_id = str(node.metadata.get("PMID") or node.metadata.get("pubid") or "UNKNOWN")
                tri_score = float(tri_result.get("evaluator_score", 0.0))

            # =====================================================
            # FIX 2: Calculate HIT (Did we find the correct doc?)
            # =====================================================
            is_naive_hit = 1 if naive_id == gt_id else 0
            is_tri_hit = 1 if tri_id == gt_id else 0

            naive_hits += is_naive_hit
            tri_hits += is_tri_hit
            total += 1

            writer.writerow([
                idx, question, gt_id,
                naive_id, tri_id,
                is_naive_hit, is_tri_hit,
                f"{naive_score:.4f}", f"{tri_score:.4f}"
            ])

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(queries)}...")
                logger.info(f"Current Accuracy -> Naive: {naive_hits / total:.2%} | Tri: {tri_hits / total:.2%}")

    logger.info(f"🎉 Final Results: Naive Accuracy: {naive_hits / total:.2%} | Tri Accuracy: {tri_hits / total:.2%}")
    return output_path
