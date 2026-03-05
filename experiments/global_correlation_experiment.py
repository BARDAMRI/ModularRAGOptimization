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


def _get_global_pool(vector_db, queries, embedding_model, k=50):
    """
    Creates a unique global document pool D from all queries and their GTs.
    """
    pool_nodes = {}
    logger.info(f"🌐 Building global pool D from {len(queries)} queries (k={k})...")

    for entry in queries:
        query = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))

        # 1. Near Query
        q_emb = np.array(embedding_model.get_text_embedding(query), dtype=np.float32)
        for hit in vector_db.retrieve(q_emb, top_k=k):
            pool_nodes[str(getattr(hit.node, 'id_', ''))] = hit.node

        # 2. Near GT
        gt_data = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["embeddings", "documents"])
        if gt_data['embeddings']:
            gt_emb = np.array(gt_data['embeddings'][0], dtype=np.float32)
            for hit in vector_db.retrieve(gt_emb, top_k=k):
                pool_nodes[str(getattr(hit.node, 'id_', ''))] = hit.node

    logger.info(f"✅ Global pool built with {len(pool_nodes)} unique documents.")
    return pool_nodes


def _generate_query_scatterplot(q_idx, df, run_dir):
    """
    Saves a scatter plot for a specific query showing LLM Score vs Distance to GT.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='llm_score', y='dist_to_gt', hue='is_gt',
                    palette={True: 'red', False: 'blue'}, alpha=0.4, s=20)

    plt.title(f"Query {q_idx}: Global Correlation Analysis")
    plt.xlabel("LLM Relevance Score")
    plt.ylabel("Distance to Query GT")
    plt.savefig(os.path.join(run_dir, f"query_{q_idx:03d}_global_scatter.png"), dpi=150)
    plt.close()


def _calculate_query_stats(q_idx, query_data, results_csv):
    """
    Logs results and returns correlation metrics for summary.
    """
    df_q = pd.DataFrame(query_data)
    df_q.to_csv(results_csv, mode='a', index=False, header=not os.path.exists(results_csv))

    corr, p_val = stats.spearmanr(df_q['llm_score'], df_q['dist_to_gt'])
    return {"query_idx": q_idx, "correlation": corr, "p_value": p_val}


def run_global_correlation_experiment(vector_db, embedding_model, num_queries=200, k=50,
                                      output_dir="results/global_exp"):
    """
    Runs the comprehensive experiment from 15.2: Query vs Global Document Pool D.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "global_results.csv")
    queries = load_qa_queries(num_queries)

    # Pre-build the global set D
    global_pool = _get_global_pool(vector_db, queries, embedding_model, k=k)

    summary_stats = []
    for q_idx, entry in enumerate(queries, start=1):
        query = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))

        # Get current query's GT embedding for distance calc
        gt_res = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["embeddings"])
        if not gt_res['embeddings']:
            continue
        current_gt_emb = np.array(gt_res['embeddings'][0], dtype=np.float32)

        logger.info(f"🔍 Testing Query {q_idx} against global pool...")
        query_data = []

        for doc_id, node in global_pool.items():
            score = gemini_score(query, node.get_content())
            if score == -1.0:
                break  # Quota hit

            dist = cosine_distance(current_gt_emb, np.array(node.embedding, dtype=np.float32))
            query_data.append({
                "query_idx": q_idx, "doc_id": doc_id, "llm_score": score,
                "dist_to_gt": dist, "is_gt": (doc_id == gt_id)
            })

        if any(d['llm_score'] == -1.0 for d in query_data):
            break

        stats_res = _calculate_query_stats(q_idx, query_data, results_csv)
        summary_stats.append(stats_res)
        _generate_query_scatterplot(q_idx, pd.DataFrame(query_data), run_dir)

    # Global summary report
    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(os.path.join(run_dir, "summary_stats.csv"), index=False)
        logger.info(f"✅ Experiment complete. All data in {run_dir}")
