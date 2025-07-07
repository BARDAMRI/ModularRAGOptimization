# main.py
from modules.model_loader import load_model, get_optimal_device
from modules.query import query_model
from modules.indexer import load_vector_db
from configurations.config import INDEX_SOURCE_URL, NQ_SAMPLE_SIZE
import sys
import termios
from scripts.evaluator import enumerate_top_documents
import os
import json
from matrics.results_logger import ResultsLogger, plot_score_distribution
import torch
from utility.logger import logger

# Import performance monitoring and caching
try:
    from utility.performance import (
        monitor_performance,
        performance_report,
        performance_monitor,
        track_performance
    )
    from utility.cache import cache_stats, clear_all_caches

    PERFORMANCE_AVAILABLE = True
    CACHE_AVAILABLE = True
except ImportError:
    logger.warning("Performance monitoring or caching not available")
    PERFORMANCE_AVAILABLE = False
    CACHE_AVAILABLE = False


    def monitor_performance(name):
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield

        return dummy_context()


    def track_performance(name=None):
        def decorator(func):
            return func

        return decorator


    def performance_report():
        print("Performance monitoring not available")


    def cache_stats():
        print("Cache monitoring not available")


    def clear_all_caches():
        print("Cache clearing not available")

# Load model and tokenizer at startup with performance monitoring
with monitor_performance("startup_model_loading"):
    tokenizer, model = load_model()


@track_performance("gpu_profiling")
def profile_gpu():
    """
    Profiles GPU utilization and memory usage.
    """
    if torch.cuda.is_available():
        print("\nGPU Utilization:")
        print(torch.cuda.memory_summary(device="cuda"))
    else:
        print("\nNo GPU detected.")


def check_device():
    """
    Checks and prints the available device for PyTorch (MPS, CUDA, or CPU).
    """
    device = get_optimal_device()
    if device.type == "mps":
        print("MPS backend is available. Using MPS for acceleration.")
    elif device.type == "cuda":
        print("CUDA backend is available. Using CUDA for acceleration.")
    else:
        print("No GPU detected. Using CPU.")
    return device


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
        print(f"\nError received during input flush: {e} ")


@track_performance("complete_query_evaluation")
def run_query_evaluation():
    """Run query evaluation with performance monitoring."""
    logger.info("Starting query evaluation...")
    profile_gpu()

    with monitor_performance("vector_db_loading"):
        logger.info("Loading external Vector DB...")
        vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

    run_mode = input("\nRun in enumeration mode? (y/n): ").strip().lower()

    if run_mode == "y":
        with monitor_performance("enumeration_mode_execution"):
            from scripts.evaluator import hill_climb_documents

            mode_choice = input("\nSelect mode: (e)numeration / (h)ill climbing: ").strip().lower()
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

                    with monitor_performance(f"query_processing_{i + 1}"):
                        if mode_choice == "h":
                            result = hill_climb_documents(i, NQ_SAMPLE_SIZE, query, vector_db, model, tokenizer,
                                                          embedding_model, top_k=5)
                            results_logger.log(result)
                        elif mode_choice == "e":
                            result = enumerate_top_documents(i, NQ_SAMPLE_SIZE, query, vector_db, embedding_model,
                                                             top_k=5)
                            results_logger.log(result)

        return
    else:
        logger.info("Entering interactive query mode...")
        device = check_device()

        while True:
            flush_input()
            user_prompt = input("\nEnter your query: ")
            if user_prompt.lower() == "exit":
                logger.info("Exiting application.")
                break

            with monitor_performance("interactive_query"):
                result = query_model(user_prompt, model, tokenizer, device, vector_db, embedding_model, max_retries=3,
                                     quality_threshold=0.5)

            if result["error"]:
                logger.error(f"Error: {result['error']}")
            else:
                logger.info(f"Question: {result['question']}, Answer: {result['answer'].strip()}")


@track_performance("results_analysis")
def run_analysis():
    """
    Runs analysis on logged results, summarizing scores and plotting score distributions.

    This function is triggered when the script is run with the `--analyze` argument.
    """
    logger.info("Running analysis on logged results...")
    logger_instance = ResultsLogger()
    logger_instance.summarize_scores()  # Print average, min, max
    plot_score_distribution()  # Show histogram of score distribution


@track_performance("development_test")
def run_development_test():
    """Quick development test with comprehensive monitoring."""
    print("Development Mode")
    print("=" * 40)

    # System info
    device = check_device()
    profile_gpu()

    # Quick model test
    print("\nTesting model capabilities...")
    from modules.model_loader import get_model_capabilities
    capabilities = get_model_capabilities(model)
    for key, value in capabilities.items():
        print(f"  {key}: {value}")

    # Quick vector DB test
    print("\nTesting vector database...")
    with monitor_performance("dev_vector_db_test"):
        vector_db, embedding_model = load_vector_db("url", INDEX_SOURCE_URL)
        print(f"Vector DB loaded: {type(vector_db).__name__}")

    # Quick query test
    print("\nTesting query processing...")
    test_query = "What is artificial intelligence?"
    with monitor_performance("dev_query_test"):
        result = query_model(test_query, model, tokenizer, device, vector_db, embedding_model)

    print(f"Test query result: {result['answer'][:100]}...")
    print(f"Score: {result['score']}")

    print("\nDevelopment test completed")


def main():
    """Main function with performance monitoring and error handling."""
    logger.info("Application started with performance monitoring.")

    try:
        # Handle command line arguments
        if "--clear-cache" in sys.argv:
            if CACHE_AVAILABLE:
                clear_all_caches()
            else:
                print("Cache functionality not available")

        if "--analyze" in sys.argv:
            run_analysis()
        elif "--performance" in sys.argv:
            if PERFORMANCE_AVAILABLE:
                performance_report()
            else:
                print("Performance monitoring not available")
        elif "--cache-stats" in sys.argv:
            if CACHE_AVAILABLE:
                cache_stats()
            else:
                print("Cache monitoring not available")
        elif "--dev" in sys.argv:
            run_development_test()
        else:
            run_query_evaluation()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        # Print final reports
        if PERFORMANCE_AVAILABLE:
            print("\nFinal Performance Report:")
            performance_report()

            # Save metrics for later analysis
            try:
                os.makedirs("results", exist_ok=True)
                performance_monitor.save_metrics("results/performance_metrics.json")
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {e}")

        if CACHE_AVAILABLE:
            print("\nFinal Cache Statistics:")
            cache_stats()


def print_help():
    """Print help information for command line usage."""
    print("RAG System - Command Line Usage")
    print("=" * 40)
    print("python main.py                 - Run interactive RAG system")
    print("python main.py --analyze       - Analyze logged results")
    print("python main.py --performance   - Show performance report")
    print("python main.py --cache-stats   - Show cache statistics")
    print("python main.py --clear-cache   - Clear all caches")
    print("python main.py --dev           - Run development test")
    print("python main.py --help          - Show this help")
    print("\nDuring interactive mode:")
    print("  Type 'exit' to quit")
    print("  Type your questions normally")


if __name__ == "__main__":
    if "--help" in sys.argv:
        print_help()
    else:
        main()
