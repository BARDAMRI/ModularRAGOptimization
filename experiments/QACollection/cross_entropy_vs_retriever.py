import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger


def calculate_document_entropy(query, document, model):
    """
    Computes the entropy-based relevance score between a query and a document.
    Currently uses the Cross-Encoder logit as a proxy for cross-entropy.

    Args:
        query (str): The search query text.
        document (str): The document text to evaluate.
        model: The model used for scoring (e.g., Cross-Encoder).

    Returns:
        float: A score representing the entropy or relevance level.
    """
    from experiments.llm_score_vs_distance import crossencoder_score
    return crossencoder_score(model, query, document)


def run_entropy_correlation_experiment(
        vector_db,
        embedding_model,
        scoring_model,
        num_queries=1000,
        output_dir="results/entropy_correlation"
):
    """
    Performs an experiment to find the correlation between cross-entropy scores and
    retriever success by comparing entropy distributions of 'good' and 'bad' queries.

    Args:
        vector_db: The vector database instance for retrieval.
        embedding_model: Model used to generate query embeddings.
        scoring_model: Model used to compute entropy/relevance.
        num_queries (int): Number of queries to sample for the experiment.
        output_dir (str): Directory path to save results and plots.

    Returns:
        None. Saves results, stats, and plots to the output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    queries = load_qa_queries(num_queries)
    results_good_baseline = []
    results_bad_baseline = []

    for q_idx, entry in enumerate(queries):
        query_text = entry.get("question")
        gt_id = str(entry.get("PMID") or entry.get("pmid") or entry.get("pubid"))

        if not query_text or not gt_id:
            continue

        query_emb = embedding_model.get_text_embedding(query_text)
        hits = vector_db.retrieve(query_emb, top_k=1)

        is_correct = False
        if hits:
            top_hit_id = str(hits[0].node.metadata.get("PMID") or hits[0].node.id_)
            if top_hit_id == gt_id:
                is_correct = True

        try:
            # Fetch Ground Truth document from DB
            gt_res = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["documents"])
            if not gt_res["documents"]:
                continue
            gt_text = gt_res["documents"][0]
        except Exception as e:
            logger.warning(f"Could not fetch GT {gt_id}: {e}")
            continue

        # Compute entropy for the GT document
        entropy_val = calculate_document_entropy(query_text, gt_text, scoring_model)

        if is_correct:
            results_good_baseline.append(entropy_val)
        else:
            results_bad_baseline.append(entropy_val)

        if q_idx % 50 == 0:
            logger.info(f"Processed {q_idx}/{len(queries)} queries")

    # Save statistical report and generate plot
    _save_statistical_report(results_good_baseline, results_bad_baseline, run_dir)
    _plot_entropy_results(results_good_baseline, results_bad_baseline, run_dir)


def _save_statistical_report(good_data, bad_data, run_dir):
    """
    Calculates and saves summary statistics for the entropy scores to a text file.

    Args:
        good_data (list): Entropy scores for 'good' queries.
        bad_data (list): Entropy scores for 'bad' queries.
        run_dir (str): Directory to save the report.
    """
    stats_path = os.path.join(run_dir, "summary_stats.txt")

    with open(stats_path, "w", encoding='utf-8') as f:
        f.write("=== Entropy Correlation Experiment Summary ===\n\n")

        for name, data in [("Good Queries (Correct Retrieval)", good_data),
                           ("Bad Queries (Incorrect Retrieval)", bad_data)]:
            if data:
                f.write(f"--- {name} ---\n")
                f.write(f"Count: {len(data)}\n")
                f.write(f"Mean Entropy Score: {np.mean(data):.4f}\n")
                f.write(f"Median Entropy Score: {np.median(data):.4f}\n")
                f.write(f"Std Dev: {np.std(data):.4f}\n")
                f.write(f"Min/Max: {np.min(data):.4f} / {np.max(data):.4f}\n\n")
            else:
                f.write(f"--- {name} ---\nNo data collected.\n\n")

    logger.info(f"Summary statistics saved to {stats_path}")


def _plot_entropy_results(good_data, bad_data, run_dir):
    """
    Generates a visual comparison of entropy distributions using jittered scatter points
    overlaid with a box-and-whisker plot.
    """
    plt.figure(figsize=(10, 7))

    # Adding jitter to x-axis for better density visualization
    x_good = np.random.normal(1, 0.04, size=len(good_data)) if good_data else []
    x_bad = np.random.normal(2, 0.04, size=len(bad_data)) if bad_data else []

    if len(good_data) > 0:
        plt.scatter(x_good, good_data, alpha=0.3, color='skyblue', s=20, label='Data Points (Good)')

    if len(bad_data) > 0:
        plt.scatter(x_bad, bad_data, alpha=0.3, color='salmon', s=20, label='Data Points (Bad)')

    # Statistical summary via boxplot
    plot_data = [d for d in [good_data, bad_data] if len(d) > 0]
    positions = [i + 1 for i, d in enumerate([good_data, bad_data]) if len(d) > 0]

    if plot_data:
        plt.boxplot(plot_data, positions=positions, widths=0.4,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor='white', alpha=0.5),
                    medianprops=dict(color='black', linewidth=2))

    plt.xticks([1, 2], ['Retriever Correct\n(Good)', 'Retriever Failed\n(Bad)'])
    plt.ylabel('Entropy-based Score (Cross-Encoder)')
    plt.title('Cross-Entropy Correlation: Good vs Bad Retrievals')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()

    plt.savefig(os.path.join(run_dir, "entropy_correlation_plot.png"), dpi=300)
    plt.close()
