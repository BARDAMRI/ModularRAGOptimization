import csv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

from utility.embedding_utils import get_text_embedding
from utility.logger import logger
from vector_db.trilateration_retriever import TrilaterationRetriever
from vector_db.vector_db_interface import VectorDBInterface


def run_retrieval_base_experiment(
        queries: list[dict],
        vector_db: VectorDBInterface,
        embed_model: HuggingFaceEmbedding,
        top_k: int,
        output_path: str,
        evaluator_model: CrossEncoder
) -> str:
    """
    Run retrieval base algorithm experiment and store results.
    Parameters
    ----------
    evaluator_model
    queries
    vector_db
    embed_model
    top_k
    output_path

    Returns
    -------
    None
    """

    logger.info(f"🚀 Starting Retrieval Base Experiment with {len(queries)} queries...")

    # Initialize trilateration retriever
    tri = TrilaterationRetriever(
        embedding_model=embed_model,
        vector_db=vector_db
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "question",
            "ground_truth_pubid",
            "naive_pubid",
            "trilateration_pubid",
            "naive_correct",
            "trilateration_correct",
            "naive_score",
            "trilateration_score"
        ])

        for idx, entry in enumerate(queries):
            question = entry.get("question")
            gt_pubid = entry.get("pubid")

            # ---- Naive retrieval ----
            q_emb = get_text_embedding(question, embed_model)
            naive_results = vector_db.retrieve(q_emb, top_k=1)
            naive_pubid = None
            naive_score = None
            if naive_results:
                naive_node = naive_results[0].node
                naive_pubid = naive_node.metadata.get("PMID") or naive_node.metadata.get("pubid")
                naive_text = getattr(naive_node, "text", None) or naive_node.get_text() if hasattr(naive_node, "get_text") else ""
                naive_score = float(evaluator_model.predict([[question, naive_text]])[0])

            # ---- Trilateration retrieval ----
            tri_result = tri.retrieve(question)
            tri_pubid = None
            tri_score = None
            if tri_result and tri_result.get("best_doc"):
                tri_doc = tri_result["best_doc"]
                tri_pubid = tri_doc.metadata.get("PMID") or tri_doc.metadata.get("pubid")
                tri_text = getattr(tri_doc, "text", None) or tri_doc.get_text() if hasattr(tri_doc, "get_text") else ""
                tri_score = float(evaluator_model.predict([[question, tri_text]])[0])

            # ---- Correctness ----
            naive_correct = int(naive_pubid == gt_pubid)
            trilateration_correct = int(tri_pubid == gt_pubid)

            writer.writerow([
                idx,
                question,
                gt_pubid,
                naive_pubid,
                tri_pubid,
                naive_correct,
                trilateration_correct,
                naive_score,
                tri_score
            ])

            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(queries)} queries...")

    logger.info(f"🎉 Retrieval experiment completed. Results saved to: {output_path}")
    return output_path
