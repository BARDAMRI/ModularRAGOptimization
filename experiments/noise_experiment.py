import numpy as np
import os
import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from configurations.config import NQ_SAMPLE_SIZE
from scripts.qa_data_set_loader import load_qa_queries
from vector_db.vector_db_interface import VectorDBInterface
from utility.logger import logger


def run_noise_experiment(
        vector_db: VectorDBInterface,
        embed_model: HuggingFaceEmbedding,
        top_k: int,
        output_path: str
):
    """
    Compare top-k retrieval results between original and noisy query vectors.
    Store results for visualization and analysis.
    """

    if vector_db.db_type.lower() != "chroma":
        logger.error("❌ Noise experiment is only supported for Chroma vector DB.")
        print(
            "❌ Noise experiment is only supported for Chroma vector DB. Please restart and choose Chroma as the vector database.")
        return

    noise_levels = [0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
    runs_per_level = 5

    try:
        test_queries = load_qa_queries(NQ_SAMPLE_SIZE)
        if not test_queries:
            logger.error("No test queries loaded.")
            return
    except Exception as e:
        logger.exception(f"❌ Failed to load QA dataset: {e}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_start_time = time.time()

    try:
        import pandas as pd

        all_results = []

        for q_idx, query in enumerate(tqdm(test_queries, desc="Running noise robustness test")):
            try:
                clean_vec = np.array(embed_model.get_text_embedding(query))
                norm = np.linalg.norm(clean_vec)
                if norm > 1e-8:
                    clean_vec /= norm
            except Exception as e:
                logger.warning(f"❌ Failed to get clean embedding for query {q_idx}: {e}")
                continue

            try:
                base_results = vector_db.retrieve(clean_vec, top_k=top_k)
                base_ids = [node.node.id_ for node in base_results]
            except Exception as e:
                logger.warning(f"❌ Failed to retrieve base results for query {q_idx + 1}: {e}")
                continue

            if not base_ids:
                logger.warning(f"⚠️ No results for base query {q_idx + 1}, skipping.")
                continue

            for epsilon in noise_levels:
                epsilon_start_time = time.time()
                logger.info(f"→ Running epsilon={epsilon} for query {q_idx + 1}")

                for run_idx in range(runs_per_level):
                    try:
                        noise = np.random.normal(0, epsilon, size=clean_vec.shape)
                        noisy_vec = clean_vec + noise
                        norm = np.linalg.norm(noisy_vec)
                        if norm > 1e-8:
                            noisy_vec /= norm

                        noisy_results = vector_db.retrieve(noisy_vec, top_k=top_k)
                        noisy_ids = [node.node.id_ for node in noisy_results]

                        if not noisy_ids:
                            logger.warning(f"⚠️ No results for noisy query (ε={epsilon}, run={run_idx})")
                            continue

                        changed = sum(1 for a, b in zip(base_ids, noisy_ids) if a != b)
                        percent_changed = changed / top_k if top_k else 0.0

                        all_results.append({
                            "query_index": q_idx + 1,
                            "query": query,
                            "epsilon": round(epsilon, 5),
                            "run": run_idx,
                            "changed_top_k": changed,
                            "percent_changed": round(percent_changed, 4)
                        })

                    except Exception as e:
                        logger.warning(f"❌ Error during noisy query (ε={epsilon}, run={run_idx}): {e}")
                        continue

                duration = round(time.time() - epsilon_start_time, 2)
                logger.info(f"✔️ Finished ε={epsilon} for query {q_idx + 1} in {duration} sec")

        df = pd.DataFrame(all_results)
        df_summary = df.groupby(["query_index", "query", "epsilon"]).agg({
            "changed_top_k": "mean",
            "percent_changed": "mean"
        }).reset_index()

        # Compute average metrics per epsilon value
        df_avg_per_epsilon = df.groupby("epsilon").agg({
            "changed_top_k": "mean",
            "percent_changed": "mean"
        }).reset_index()

        output_xlsx_path = output_path.replace(".csv", ".xlsx")
        with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="all_runs", index=False)
            df_summary.to_excel(writer, sheet_name="averaged", index=False)
            df_avg_per_epsilon.to_excel(writer, sheet_name="epsilon_avg", index=False)
            create_noise_chart_xlsx(writer, noise_levels)

    except Exception as e:
        logger.exception(f"❌ Failed to write results to file: {e}")
        print(f"❌ Failed to write results to file: {e}")
        return

    total_duration = round(time.time() - total_start_time, 2)
    logger.info(f"✅ Noise experiment completed in {total_duration} seconds")
    logger.info(f"📄 Results saved to: {output_xlsx_path}")


def create_noise_chart_xlsx(writer, epsilons):
    sheet_name = "averaged_graph"
    wb = writer.book
    ws = wb.create_sheet(title=sheet_name)

    ws.append(["epsilon", "percent_range"])

    percent_values = [round(p, 2) for p in np.linspace(0, 1.0, 11)]
    max_len = max(len(epsilons), len(percent_values))

    for i in range(max_len):
        eps = epsilons[i] if i < len(epsilons) else None
        pct = percent_values[i] if i < len(percent_values) else None
        ws.append([eps, pct])
