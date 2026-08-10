import os
from datetime import datetime
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import ACTIVE_QA_DATASET, NQ_SAMPLE_SIZE, RETRIEVER_TOP_K
from experiments.retrieval_experiment import run_retrieval_base_experiment
from scripts.qa_data_set_loader import load_qa_queries
from utility.logger import logger
from utility.user_interface import validate_prerequisites, print_section_header, PROJECT_PATH, prompt_with_validation
from vector_db.vector_db_interface import VectorDBInterface


def run_retrieval_base_algorithm_experiment(
        vector_db: VectorDBInterface,
        embedding_model: HuggingFaceEmbedding,
        evaluator_model: torch.nn.Module
):
    """Run the Retrieval Baseline experiment with the specified vector database and embedding model.
    This experiment evaluates the performance of a simple retrieval-based approach on a sample of queries from the active QA dataset.
    This function performs the following steps:
    1. Validates prerequisites for the experiment.
    2. Loads a sample of queries from the specified QA dataset.
    3. Prompts the user for an output filename, with a default suggestion based on the vector database and embedding model.
    4. Executes the retrieval baseline experiment and saves results to the specified CSV file.
    """
    # Validate prerequisites
    if not validate_prerequisites("retrieval_base", ACTIVE_QA_DATASET, vector_db):
        return

    print_section_header("🧪 SIMPLE RETRIEVAL BASELINE EXPERIMENT")

    # ---------------------------
    # 1. Load queries generically
    # ---------------------------
    try:
        queries = load_qa_queries(NQ_SAMPLE_SIZE)
        print(f"📁 Loaded {len(queries)} queries for experiment.")
    except Exception as e:
        print(f"❌ Failed to load QA queries: {e}")
        logger.error(f"Failed to load queries: {e}")
        return

    # ---------------------------
    # 2. Prepare output filename
    # ---------------------------
    db_type = vector_db.db_type
    distance_metric = vector_db.distance_metric
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    default_filename = f"{db_type}_{distance_metric}_top_{RETRIEVER_TOP_K}_results_retrieval_base_{timestamp}.csv"
    filename = prompt_with_validation(
        f"Enter output CSV filename (press Enter for default: {default_filename}):\n",
        lambda s: s == "" or s.endswith('.csv'),
        default=default_filename
    )

    if not filename or not filename.endswith('.csv'):
        logger.warning("Invalid filename. Using default.")
        filename = default_filename

    output_dir = os.path.join(PROJECT_PATH, "results", "retrieval_base_experiment")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # ---------------------------
    # 3. Run experiment
    # ---------------------------
    run_retrieval_base_experiment(
        queries=queries,
        vector_db=vector_db,
        embed_model=embedding_model,
        top_k=RETRIEVER_TOP_K,
        output_path=output_path,
        evaluator_model=evaluator_model
    )

    print("✅ Retrieval Baseline experiment completed.")
