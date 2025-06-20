# main.py
from modules.model_loader import load_model
from modules.query import query_model
from modules.indexer import load_vector_db
from config import INDEX_SOURCE_URL
import sys
import termios
from scripts.evaluator import enumerate_top_documents
import os
import json
from config import NQ_SAMPLE_SIZE
from matrics.results_logger import ResultsLogger, plot_score_distribution
import torch
from sentence_transformers import SentenceTransformer
from utility.logger import logger  # Import logger from utility/logger.py

tokenizer, model = load_model()


def profile_gpu():
    """
    Profiles GPU utilization and memory usage.
    """
    if torch.cuda.is_available():
        print("\n> GPU Utilization:")
        print(torch.cuda.memory_summary(device="cuda"))
    else:
        print("\n> No GPU detected.")


def check_device():
    """
    Checks and prints the available device for PyTorch (MPS, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        print("MPS backend is available. Using MPS for acceleration.")
        return "mps"
    elif torch.cuda.is_available():
        print("CUDA backend is available. Using CUDA for acceleration.")
        return "cuda"
    else:
        print("No GPU detected. Using CPU.")
        return "cpu"


# Example usage:
device = check_device()


def flush_input():
    """
    Flushes accidental keyboard input from the terminal buffer before reading input.

    This function ensures that any unintended input in the terminal buffer is cleared
    to avoid interference with subsequent user input.

    Raises:
        Exception: If an error occurs during the flush operation.
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception as e:
        print(f"\n> Error received during input flush: {e} ")


def run_query_evaluation():
    logger.info("Starting query evaluation...")
    profile_gpu()

    logger.info("Loading external Vector DB...")
    vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

    profile_gpu()

    run_mode = input("\n🛠️ Run in enumeration mode? (y/n): ").strip().lower()
    if run_mode == "y":
        from scripts.evaluator import hill_climb_documents

        mode_choice = input("\n🧪 Select mode: (e)numeration / (h)ill climbing: ").strip().lower()
        results_logger = ResultsLogger(top_k=5, mode="hill" if mode_choice == "h" else "enum")
        nq_file_path = "data/user_query_datasets/natural-questions-master/nq_open/NQ-open.dev.jsonl"

        if not os.path.exists(nq_file_path):
            logger.error(f"NQ file not found at: {nq_file_path}")
            return

        with open(nq_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= NQ_SAMPLE_SIZE:
                    break
                data = json.loads(line)
                query = data.get("question")
                logger.info(f"Running for NQ Query #{i + 1}: {query}")

                if mode_choice == "h":
                    result = hill_climb_documents(i, NQ_SAMPLE_SIZE, query, vector_db, model, tokenizer,
                                                  embedding_model, top_k=5)
                    results_logger.log(result)
                elif mode_choice == "e":
                    result = enumerate_top_documents(i, NQ_SAMPLE_SIZE, query, vector_db, embedding_model, top_k=5)
                    results_logger.log(result)

                profile_gpu()
        return
    else:
        logger.info("Entering interactive query mode...")
        while True:
            flush_input()
            user_prompt = input("\n💬 Enter your query: ")
            if user_prompt.lower() == "exit":
                logger.info("Exiting application.")
                break

            result = query_model(user_prompt, model, tokenizer, device, vector_db, embedding_model, max_retries=3, quality_threshold=0.5)
            if result["error"]:
                logger.error(f"Error: {result['error']}")
            else:
                logger.info(f"Question: {result['question']}, Answer: {result['answer'].strip()}")

            profile_gpu()


def run_analysis():
    """
    Runs analysis on logged results, summarizing scores and plotting score distributions.

    This function is triggered when the script is run with the `--analyze` argument.
    """
    logger.info("Running analysis on logged results...")
    logger_instance = ResultsLogger()
    logger_instance.summarize_scores()  # Print average, min, max
    plot_score_distribution()  # Show histogram of score distribution


if __name__ == "__main__":
    logger.info("Application started.")
    if "--analyze" in sys.argv:
        run_analysis()
    else:
        run_query_evaluation()
