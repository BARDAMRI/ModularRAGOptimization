import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from experiments.llm_score_vs_distance import gemini_score
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from vector_db.trilateration_retriever import cosine_distance


def _save_experiment_config(run_dir, params):
    """
    Saves the experiment metadata and parameters for full reproducibility.
    """
    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(params, f, indent=4)


def _log_pool_manifest(run_dir, pool_nodes):
    """
    Records exactly which documents formed the global set D and their IDs.
    """
    manifest_data = [{"doc_id": doc_id, "text_snippet": node.get_content()[:100]}
                     for doc_id, node in pool_nodes.items()]
    pd.DataFrame(manifest_data).to_csv(os.path.join(run_dir, "pool_manifest.csv"), index=False)


def _check_rag_baseline(vector_db, query_emb, gt_id):
    """
    Determines if the standard retriever finds the GT at Rank 1.
    """
    initial_retrieval = vector_db.retrieve(query_emb, top_k=1)
    if not initial_retrieval:
        return True
    top_hit_id = str(getattr(initial_retrieval[0].node, 'id_', ''))
    return top_hit_id != str(gt_id)


def _log_failure_analysis(run_dir, q_idx, query, query_df):
    """
    Documents cases where non-GT documents scored as high or higher than the GT.
    """
    gt_row = query_df[query_df['is_gt']]
    if gt_row.empty: return

    gt_score = gt_row['llm_score'].iloc[0]
    outliers = query_df[(~query_df['is_gt']) & (query_df['llm_score'] >= gt_score)]

    if outliers.empty: return

    file_path = os.path.join(run_dir, "detailed_failures.md")
    mode = "a" if os.path.exists(file_path) else "w"

    with open(file_path, mode, encoding="utf-8") as f:
        if mode == "w": f.write("# 🕵️ Global Pool Failure Analysis\n\n")
        f.write(f"### Query {q_idx}: {query}\n")
        f.write(f"**GT Score:** {gt_score} | **Outliers found:** {len(outliers)}\n\n")
        f.write("| Doc ID | Score | Dist to GT | Status |\n| --- | --- | --- | --- |\n")
        for _, row in outliers.sort_values(by='llm_score', ascending=False).head(10).iterrows():
            f.write(
                f"| {row['doc_id']} | {row['llm_score']} | {row['dist_to_gt']:.4f} | {'Higher' if row['llm_score'] > gt_score else 'Tie'} |\n")
        f.write("\n---\n\n")


def _generate_query_scatterplot(q_idx, df, run_dir):
    """
    Saves a scatter plot for a specific query showing LLM Score vs Distance to GT.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='llm_score', y='dist_to_gt', hue='is_gt',
                    palette={True: 'red', False: 'blue'}, alpha=0.4, s=25)
    plt.title(f"Query {q_idx}: Global Correlation Analysis")
    plt.xlabel("LLM Relevance Score (0.1-1.0)")
    plt.ylabel("Distance to GT Embedding")
    plt.savefig(os.path.join(run_dir, f"query_{q_idx:03d}_global_scatter.png"), dpi=150)
    plt.close()


def _generate_final_summary_plot(summary_df, run_dir):
    """
    Generates a summary plot sorted by correlation and colors RAG failures.
    """
    summary_df = summary_df.sort_values(by='correlation', ascending=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=summary_df, x='query_idx', y='correlation', hue='rag_failed',
                palette={True: 'orange', False: 'green'}, dodge=False)
    plt.title("Spearman Correlation per Query (Green=RAG Success, Orange=RAG Failed)")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "global_summary_overview.png"), dpi=200)
    plt.close()


def _get_global_pool(vector_db, queries, embedding_model, k=50):
    """
    Creates a unique global document pool D from all queries and their GTs.
    """
    pool_nodes = {}
    logger.info(f"🌐 Building global pool D from {len(queries)} queries (k={k})...")

    for entry in queries:
        query = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))

        q_emb = np.array(embedding_model.get_text_embedding(query), dtype=np.float32)
        for hit in vector_db.retrieve(q_emb, top_k=k):
            pool_nodes[str(getattr(hit.node, 'id_', ''))] = hit.node

        gt_data = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["embeddings", "documents"])
        if gt_data['embeddings']:
            gt_emb = np.array(gt_data['embeddings'][0], dtype=np.float32)
            for hit in vector_db.retrieve(gt_emb, top_k=k):
                pool_nodes[str(getattr(hit.node, 'id_', ''))] = hit.node

    logger.info(f"✅ Global pool built with {len(pool_nodes)} unique documents.")
    return pool_nodes


def run_global_correlation_experiment(vector_db, embedding_model, num_queries=200, k=50,
                                      output_dir="results/global_exp"):
    """
    Executes the 15.2 correlation experiment with full metadata logging, resume capability, and failure analysis.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    _save_experiment_config(run_dir, {
        "timestamp": timestamp, "num_queries": num_queries, "k": k, "model": "gemini-2.0-flash"
    })

    results_csv = os.path.join(run_dir, "global_results.csv")
    queries = load_qa_queries(num_queries)

    global_pool = _get_global_pool(vector_db, queries, embedding_model, k=k)
    _log_pool_manifest(run_dir, global_pool)

    completed_queries = set()
    if os.path.exists(results_csv):
        completed_queries = set(pd.read_csv(results_csv)['query_idx'].unique())

    summary_stats = []
    for q_idx, entry in enumerate(queries, start=1):
        if q_idx in completed_queries:
            continue

        query_text = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))

        gt_res = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["embeddings"])
        if not gt_res['embeddings']: continue
        current_gt_emb = np.array(gt_res['embeddings'][0], dtype=np.float32)

        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        rag_failed = _check_rag_baseline(vector_db, query_emb, gt_id)

        logger.info(f"🔍 Testing Query {q_idx}/{num_queries} against pool of {len(global_pool)} docs...")
        query_data = []

        for doc_id, node in global_pool.items():
            score = gemini_score(query_text, node.get_content())
            if score == -1.0: break

            dist = cosine_distance(current_gt_emb, np.array(node.embedding, dtype=np.float32))
            query_data.append({
                "query_idx": q_idx, "doc_id": doc_id, "llm_score": score,
                "dist_to_gt": dist, "is_gt": (doc_id == gt_id), "rag_failed": rag_failed
            })

        if any(d['llm_score'] == -1.0 for d in query_data):
            logger.error("🛑 Stopping experiment due to fatal API error.")
            break

        query_df = pd.DataFrame(query_data)
        query_df.to_csv(results_csv, mode='a', index=False, header=not os.path.exists(results_csv))

        _log_failure_analysis(run_dir, q_idx, query_text, query_df)
        _generate_query_scatterplot(q_idx, query_df, run_dir)

        corr, _ = stats.spearmanr(query_df['llm_score'], query_df['dist_to_gt'])
        summary_stats.append({"query_idx": q_idx, "correlation": corr, "rag_failed": rag_failed})

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(run_dir, "summary_stats.csv"), index=False)
        _generate_final_summary_plot(summary_df, run_dir)
        logger.info(f"✅ Full experiment logged in {run_dir}")
