import numpy as np
import csv
import os
from typing import List, Set
import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from configurations.config import NQ_SAMPLE_SIZE  # Assuming this config exists and is correctly set
from scripts.qa_data_set_loader import load_qa_queries  # Assuming this script exists
from vector_db.vector_db_interface import VectorDBInterface

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculates the Jaccard Similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        # If both sets are empty, similarity is 0 (or 1 depending on definition, but 0 is common here)
        return 0.0
    return intersection / union


def calculate_recall_at_k(reference_set: Set[str], retrieved_list: List[str]) -> float:
    """
    Calculates Recall@k: how many items from the reference_set are found in the retrieved_list.
    """
    if not reference_set:  # If the reference set is empty, Recall is 1.0 (nothing to miss)
        return 1.0
    retrieved_set = set(retrieved_list)
    relevant_retrieved = len(reference_set.intersection(retrieved_set))
    return relevant_retrieved / len(reference_set)


def run_noise_experiment(
        vector_db: VectorDBInterface,
        embed_model: HuggingFaceEmbedding,
        top_k: int,
        output_csv_path: str):
    """
    Evaluate the robustness of the vector database to additive noise in query embeddings.
    Uses Jaccard Similarity and Recall@k to measure consistency.

    Args:
        vector_db (VectorDBInterface): An initialized vector database instance.
        embed_model (HuggingFaceEmbedding): The embedding model used for queries.
        top_k (int): The number of top results to retrieve.
        output_csv_path (str): The path to save the experiment results CSV.
    """

    # Define epsilon values to test (non-linear steps for fine-grained analysis)
    noise_levels = [0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
    num_runs_per_epsilon = 5  # Number of runs for each epsilon to get a stable average

    # Load test queries from QA collection
    try:
        # NQ_SAMPLE_SIZE should be defined in configurations.config
        test_queries = load_qa_queries(NQ_SAMPLE_SIZE)
        if not test_queries:
            logger.error("No test queries loaded. Exiting experiment.")
            return
        logger.info(f"Loaded {len(test_queries)} test queries.")
    except Exception as e:
        logger.error(f"Failed to load QA dataset: {e}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Initial Consistency Check (Text Query vs. Embedding Query via Chroma ANN) ---
    logger.info("\n--- Initial Consistency Check (Text Query vs. Embedding Query via Chroma ANN) ---")

    # Use the first query from the set for this sample check
    sample_query = test_queries[0]
    # Ensure embedding generation is deterministic by global seed setting
    sample_embedding = np.array(embed_model.get_text_embedding(sample_query))

    logger.info(f"Sample Query: '{sample_query}'")
    logger.info(f"Sample Embedding shape: {sample_embedding.shape}")
    logger.info(f"Sample Embedding norm: {np.linalg.norm(sample_embedding):.4f}")

    # Retrieve using text query (uses collection.query with query_texts)
    text_results_sample = vector_db.retrieve(sample_query, top_k=top_k)
    text_ids_sample = set([node.node.id_ for node in text_results_sample])
    text_scores_sample = {node.node.id_: node.score for node in text_results_sample}

    # Retrieve using embedding query (now uses collection.query with query_embeddings)
    embedding_results_sample = vector_db.retrieve(sample_embedding, top_k=top_k)
    embedding_ids_sample = set([node.node.id_ for node in embedding_results_sample])
    embedding_scores_sample = {node.node.id_: node.score for node in embedding_results_sample}

    # Log consistency check results
    logger.info(f"Text query results count: {len(text_results_sample)}")
    logger.info(f"Embedding query results count: {len(embedding_results_sample)}")

    if not text_results_sample or not embedding_results_sample:
        logger.warning("One or both sample retrievals returned no results. Check DB population or embedding issues.")
    else:
        logger.info(f"Text IDs (sorted): {sorted(list(text_ids_sample))}")
        logger.info(f"Embedding IDs (sorted): {sorted(list(embedding_ids_sample))}")

        jaccard_val = calculate_jaccard_similarity(text_ids_sample, embedding_ids_sample)
        # Note: Recall is calculated as (Embed vs Text) meaning, how many of the Text IDs were retrieved by Embed ID.
        recall_val = calculate_recall_at_k(text_ids_sample, list(embedding_ids_sample))
        logger.info(f"Jaccard Similarity (Text vs Embed): {jaccard_val:.4f}")
        logger.info(f"Recall@k (Embed vs Text): {recall_val:.4f}")

        # Calculate score differences for common IDs in this sample check
        common_ids_sample = text_ids_sample.intersection(embedding_ids_sample)
        if common_ids_sample:
            sample_diffs = [
                abs(text_scores_sample.get(id, 0) - embedding_scores_sample.get(id, 0))
                for id in common_ids_sample
            ]
            logger.info(f"Average Score Diff for common IDs (Sample): {np.mean(sample_diffs):.4f}")
        else:
            logger.info("No common IDs between text and embedding sample queries.")

        # Log scores for deeper inspection
        logger.debug(f"Distances (Text Query Scores): {[f'{s:.4f}' for s in text_scores_sample.values()]}")
        logger.debug(f"Distances (Embedding Query Scores): {[f'{s:.4f}' for s in embedding_scores_sample.values()]}")
    logger.info("--- End Initial Consistency Check ---\n")

    # Open the CSV file to record experiment results
    with open(output_csv_path, mode="w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["query", "epsilon", "jaccard_similarity", "recall_at_k", "avg_score_diff", "std_score_diff"])

        # Iterate over all test queries
        for query_idx, query in enumerate(tqdm(test_queries, desc="Running noise robustness test")):
            # Step 1: Establish a "stable" baseline for the current query
            # Ensure query embedding generation is deterministic (controlled by global seed)
            query_vec_base = np.array(embed_model.get_text_embedding(query))
            # Re-normalize after embedding if not already done by get_text_embedding
            if np.linalg.norm(query_vec_base) != 0:
                query_vec_base = query_vec_base / np.linalg.norm(query_vec_base)

            # Retrieve baseline results for reference (epsilon=0) - run once per query
            # This call MUST be deterministic for the noise experiment to be valid.
            base_results_for_ref = vector_db.retrieve(query_vec_base, top_k=top_k)
            base_ids_reference = set([node.node.id_ for node in base_results_for_ref])
            base_scores_reference = {node.node.id_: node.score for node in base_results_for_ref}

            # Check if baseline retrieval is successful for this query
            if not base_ids_reference:
                logger.warning(f"Baseline retrieval for query '{query}' returned no results. Skipping this query.")
                continue  # Skip this query if no baseline results

            # Iterate over noise levels
            for epsilon in noise_levels:
                jaccard_scores_epsilon = []
                recall_scores_epsilon = []
                score_diffs_epsilon_flat = []  # To store all score differences across runs for std dev

                # Run multiple times for each epsilon to get a stable average
                for run in range(num_runs_per_epsilon):
                    # Add Gaussian noise to the query vector
                    # np.random.normal is deterministic if np.random.seed is set globally at script start.
                    noise = np.random.normal(0, epsilon, size=query_vec_base.shape)
                    noisy_query_vec = query_vec_base + noise

                    # Re-normalize after adding noise to maintain vector length if needed for metric
                    if np.linalg.norm(noisy_query_vec) != 0:
                        noisy_query_vec = noisy_query_vec / np.linalg.norm(noisy_query_vec)
                    else:
                        logger.warning(
                            f"Noisy query vector for query '{query}' (epsilon={epsilon}, run={run}) has zero norm after noise. Skipping normalization.")

                    # Retrieve using the noisy vector
                    # This call also relies on the determinism of vector_db.retrieve
                    noisy_results = vector_db.retrieve(noisy_query_vec, top_k=top_k)
                    noisy_ids = [node.node.id_ for node in noisy_results]
                    noisy_scores = {node.node.id_: node.score for node in noisy_results}

                    # Check if noisy retrieval is successful
                    if not noisy_ids:
                        logger.warning(
                            f"Noisy retrieval for query '{query}' (epsilon={epsilon}, run={run}) returned no results. "
                            f"This run's Jaccard/Recall will be 0 if baseline has results."
                        )
                        # Ensure lists are populated with default 0 if no results to avoid errors in np.mean
                        jaccard_scores_epsilon.append(0.0)
                        recall_scores_epsilon.append(0.0)
                        continue  # Skip score diff calculation for this run if no results

                    # Calculate Jaccard Similarity
                    jaccard = calculate_jaccard_similarity(base_ids_reference, set(noisy_ids))
                    jaccard_scores_epsilon.append(jaccard)

                    # Calculate Recall@k
                    recall = calculate_recall_at_k(base_ids_reference, noisy_ids)
                    recall_scores_epsilon.append(recall)

                    # Calculate score differences for common IDs
                    common_ids = base_ids_reference.intersection(set(noisy_ids))
                    if common_ids:
                        diffs = [
                            abs(base_scores_reference.get(id, 0) - noisy_scores.get(id, 0))
                            for id in common_ids
                        ]
                        score_diffs_epsilon_flat.extend(diffs)
                    else:
                        logger.debug(f"No common IDs for query '{query}', epsilon={epsilon}, run={run}.")

                # Calculate averages and standard deviations for this epsilon
                avg_jaccard = np.mean(jaccard_scores_epsilon)
                avg_recall = np.mean(recall_scores_epsilon)

                avg_score_diff = np.mean(score_diffs_epsilon_flat) if score_diffs_epsilon_flat else 0.0
                std_score_diff = np.std(score_diffs_epsilon_flat) if score_diffs_epsilon_flat else 0.0

                writer.writerow([
                    query,
                    round(epsilon, 5),
                    round(avg_jaccard, 4),
                    round(avg_recall, 4),
                    round(avg_score_diff, 4),
                    round(std_score_diff, 4)
                ])
                # Log progress to console
                logger.info(
                    f"Query: '{query_idx + 1}/{len(test_queries)}', Epsilon: {epsilon:.5f}, Jaccard: {avg_jaccard:.4f}, Recall: {avg_recall:.4f}, Avg Score Diff: {avg_score_diff:.4f}")

    logger.info(f"✅ Noise robustness experiment completed. Results saved to: {output_csv_path}")
