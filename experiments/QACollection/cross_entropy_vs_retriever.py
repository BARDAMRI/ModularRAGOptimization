import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from experiments.llm_score_vs_distance import crossencoder_score
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger


def calculate_document_entropy(query, document, model):
    """
    Computes the entropy-based relevance score between a query and a document.
    Currently uses the Cross-Encoder logit as a proxy for cross-entropy.

    Args:
        query (str): The search query text.
        document (str): The document text to evaluate.
        model: The model used for scoring (e.g., Cross-Encoder or LLM).

    Returns:
        float: A score representing the entropy or relevance level.
    """
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
        scoring_model: Model used to compute entropy/relevance (e.g., Cross-Encoder).
        num_queries (int): Number of queries to sample for the experiment.
        output_dir (str): Directory path to save results and plots.

    Returns:
        None. Saves a combined scatter and box plot to the output directory.
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
            gt_res = vector_db.collection.get(where={"PMID": int(gt_id)}, include=["documents"])
            gt_text = gt_res["documents"][0]
        except:
            continue

        entropy_val = calculate_document_entropy(query_text, gt_text, scoring_model)

        if is_correct:
            results_good_baseline.append(entropy_val)
        else:
            results_bad_baseline.append(entropy_val)

        if q_idx % 50 == 0:
            logger.info(f"Processed {q_idx}/{len(queries)} queries")

    _plot_entropy_results(results_good_baseline, results_bad_baseline, run_dir)


def _plot_entropy_results(good_data, bad_data, run_dir):
    """
    Generates a visual comparison of entropy distributions using jittered scatter points
    overlaid with a box-and-whisker plot.

    Args:
        good_data (list): List of entropy scores for queries where retriever was correct.
        bad_data (list): List of entropy scores for queries where retriever failed.
        run_dir (str): Directory path to save the generated plot.

    Returns:
        None.
    """
    plt.figure(figsize=(10, 7))

    x_good = np.random.normal(1, 0.04, size=len(good_data))
    x_bad = np.random.normal(2, 0.04, size=len(bad_data))

    plt.scatter(x_good, good_data, alpha=0.3, color='skyblue', s=20, label='Data Points (Good)')
    plt.scatter(x_bad, bad_data, alpha=0.3, color='salmon', s=20, label='Data Points (Bad)')

    plt.boxplot([good_data, bad_data], positions=[1, 2], widths=0.4,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='white', alpha=0.5),
                medianprops=dict(color='black', linewidth=2))

    plt.xticks([1, 2], ['Retriever Correct\n(Good)', 'Retriever Failed\n(Bad)'])
    plt.ylabel('Entropy-based Score')
    plt.title('Entropy Distribution vs. Retrieval Success')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(run_dir, "entropy_comparison_plot.png"), dpi=300)
    plt.close()
