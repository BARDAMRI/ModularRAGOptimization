import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from llama_index.core.schema import TextNode
from scipy import stats

from experiments.llm_score_vs_distance import gemini_score, sample_random_docs_by_pmid
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance


def log_failure_to_markdown(run_dir, q_idx, query, gt_node, outlier_node, gt_score, outlier_score):
    """
    Records cases where a non-ground-truth document received a score equal to or higher than the ground truth.

    Args:
        run_dir (str): Directory where the report will be saved.
        q_idx (int): The index of the current query in the experiment.
        query (str): The text of the query.
        gt_node (TextNode): The node containing the ground truth document and metadata.
        outlier_node (TextNode): The node containing the non-GT document that scored high.
        gt_score (float): The score assigned by the LLM to the GT document.
        outlier_score (float): The score assigned by the LLM to the outlier document.

    Returns:
        None
    """
    file_path = os.path.join(run_dir, "failure_analysis.md")
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# ⚠️ LLM Failure Analysis - Outlier Detection\n\n")

    with open(file_path, "a", encoding="utf-8") as f:
        gt_id = getattr(gt_node, 'id_', 'N/A')
        out_id = getattr(outlier_node, 'id_', 'N/A')

        f.write(f"### Query {q_idx}: {query}\n")
        f.write(f"**Insight:** Outlier (Score: {outlier_score}) >= GT (Score: {gt_score})\n\n")
        f.write("| Feature | Ground Truth (GT) | Outlier Document |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **ID** | {gt_id} | {out_id} |\n")
        f.write(f"| **LLM Score** | **{gt_score}** | **{outlier_score}** |\n")
        f.write(f"| **Text** | {gt_node.text[:600]}... | {outlier_node.text[:600]}... |\n\n")
        f.write("---\n\n")


def generate_correlation_plot(df, run_dir):
    """
    Creates a scatter plot illustrating the correlation between vector distance (from GT) and LLM relevance scores.

    Args:
        df (pd.DataFrame): Dataframe containing 'dist_to_gt', 'llm_score', and 'is_gt' columns.
        run_dir (str): Directory where the plot image will be saved.

    Returns:
        None
    """
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    clean_df = df.dropna(subset=['dist_to_gt', 'llm_score'])
    if len(clean_df) < 2:
        return

    r_coef, _ = stats.pearsonr(clean_df['dist_to_gt'], clean_df['llm_score'])

    sns.scatterplot(data=df[~df['is_gt']], x='dist_to_gt', y='llm_score', alpha=0.5, label="Others")
    sns.scatterplot(data=df[df['is_gt']], x='dist_to_gt', y='llm_score', s=150, color='red', marker='*', label="GT")
    sns.regplot(data=df, x='dist_to_gt', y='llm_score', scatter=False, color='orange', label=f"Trend (r={r_coef:.2f})")

    plt.title("Correlation: Vector Distance from GT vs LLM Score")
    plt.xlabel("Distance to GT Embedding")
    plt.ylabel("Normalized LLM Score (0.1-1.0)")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "correlation_plot.png"), dpi=200)
    plt.close()


def run_llm_relevance_experiment(vector_db, embedding_model, num_queries=200, output_dir="results/llm_exp"):
    """
    Orchestrates a three-part relevance experiment with a checkpoint mechanism to allow resuming.
    Saves results incrementally to results.csv and llm_failures.csv.

    Args:
        vector_db: The vector database instance.
        embedding_model: The embedding model instance for retrieval.
        num_queries (int): Total number of queries to process.
        output_dir (str): Directory for experiment output.

    Returns:
        pd.DataFrame: The final aggregated results dataframe.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "results.csv")
    failures_csv = os.path.join(run_dir, "llm_failures.csv")

    completed_queries = set()
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        if not existing_df.empty:
            completed_queries = set(existing_df['query_idx'].unique())
            logger.info(
                f"🔄 Found existing results. Resuming from query {max(completed_queries) + 1 if completed_queries else 1}")

    queries = load_qa_queries(num_queries)

    for q_idx, entry in enumerate(queries, start=1):
        if q_idx in completed_queries:
            continue

        query = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))
        logger.info(f"🚀 Processing Query {q_idx}: {query[:50]}...")

        try:
            #  Fetch GT document and embedding
            gt_data = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["documents", "embeddings"])
            if not gt_data['documents']:
                logger.warning(f"⚠️ GT {gt_id} not found.")
                continue

            gt_node = TextNode(text=gt_data['documents'][0], embedding=np.array(gt_data['embeddings'][0]))
            gt_node.id_ = gt_id
            # Score the GT document with Gemini.
            gt_score = gemini_score(query, gt_node.text)

            if gt_score == -1.0:
                #  Fatal error, likely quota exceeded
                break

        except Exception as e:
            logger.error(f"❌ Error fetching GT {gt_id}: {e}")
            continue

        #  Retrieve candidate documents from closest embeddings and random samples.
        query_emb = np.array(embedding_model.get_text_embedding(query), dtype=np.float32)
        closest_hits = vector_db.retrieve(query_emb, top_k=20)
        random_nodes = sample_random_docs_by_pmid(vector_db, 10)
        all_test_nodes = [hit.node for hit in closest_hits] + random_nodes

        current_query_results = []
        current_query_failures = []

        for node in all_test_nodes:
            node_meta = getattr(node, 'metadata', {}) or {}
            node_id = str(
                node_meta.get("PMID") or node_meta.get("pmid") or node_meta.get("pubid") or getattr(node, 'id_',
                                                                                                    'unknown'))
            is_gt = (node_id == gt_id)

            doc_text = node.get_content() if hasattr(node, "get_content") else node.text
            # score the document with Gemini
            time.sleep(1.0)
            current_score = gemini_score(query, doc_text)

            if current_score == -1.0:
                #  Fatal error, likely quota exceeded
                break

            #  Compute distance between document to GT embedding
            dist_to_gt = cosine_distance(
                np.array(gt_node.embedding, dtype=np.float32),
                np.array(node.embedding, dtype=np.float32)
            )

            current_query_results.append({
                "query_idx": q_idx,
                "is_gt": is_gt,
                "llm_score": current_score,
                "dist_to_gt": dist_to_gt,
                "doc_id": node_id
            })

            if not is_gt and current_score >= gt_score:
                log_failure_to_markdown(run_dir, q_idx, query, gt_node, node, gt_score, current_score)
                current_query_failures.append({
                    "query_idx": q_idx,
                    "query": query,
                    "gt_doc_id": gt_id,
                    "outlier_doc_id": node_id,
                    "gt_score": gt_score,
                    "outlier_score": current_score,
                    "score_delta": round(current_score - gt_score, 2),
                    "dist_to_gt": dist_to_gt,
                    "gt_text": gt_node.text,
                    "outlier_text": doc_text
                })

        if current_query_results:
            pd.DataFrame(current_query_results).to_csv(
                results_csv,
                mode='a',
                index=False,
                header=not os.path.exists(results_csv)
            )

        if current_query_failures:
            pd.DataFrame(current_query_failures).to_csv(
                failures_csv,
                mode='a',
                index=False,
                header=not os.path.exists(failures_csv)
            )

        if any(r['llm_score'] == -1.0 for r in current_query_results) or gt_score == -1.0:
            logger.error("🛑 Stopping experiment due to fatal quota error.")
            break

    if os.path.exists(results_csv):
        final_df = pd.read_csv(results_csv)
        generate_correlation_plot(final_df, run_dir)
        logger.info(f"✅ Experiment complete. Results in {run_dir}")
        return final_df

    return pd.DataFrame()
