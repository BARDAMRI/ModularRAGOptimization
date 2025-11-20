# main.py
import os
import sys
import warnings

import torch

from matrics.results_logger import ResultsLogger, plot_score_distribution
from scripts.qa_data_set_loader import download_qa_dataset
from utility.logger import logger
from utility.performance import (performance_report, performance_monitor, track_performance)
from utility.user_interface import display_main_menu, show_system_info, run_interactive_mode, run_evaluation_mode, \
    run_development_test, startup_initialization, setup_vector_database, display_startup_banner, ask_yes_no, \
    handle_command_line_args, show_exit_message, show_error_message, \
    confirm_reset_vector_db, show_vector_db_success, show_performance_summary_notice, show_experiments_menu, \
    run_retrieval_base_algorithm_experiment, cache_stats

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.backends.mps.is_available():
    torch.backends.mps.allow_tf32 = False
    logger.info("🔧 MPS configured for compatibility")

# Suppress FutureWarning specifically from huggingface_hub
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0.*",
    category=FutureWarning,
    module='huggingface_hub'
)

# Import performance monitoring with fallbacks
try:

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
        logger.info("Performance monitoring not available")


    def cache_stats():
        logger.info("Cache monitoring not available")


    def clear_all_caches():
        logger.info("Cache clearing not available")

# Global variables for model and vector DB
tokenizer = None
model = None
vector_db = None
embedding_model = None
device = None
storing_method = None
source_path = None
distance_metric = None


@track_performance("results_analysis")
def run_analysis():
    """Run analysis on logged results."""
    logger.info("\n📈 RESULTS ANALYSIS")
    logger.info("=" * 25)

    logger.info("Running analysis on logged results...")
    try:
        logger_instance = ResultsLogger()
        logger_instance.summarize_scores()
        plot_score_distribution()
        logger.info("✅ Analysis completed")
    except Exception as e:
        logger.info(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis error: {e}")


def download_dataset():
    """Download the QA dataset."""

    logger.info("\n📥 Downloading QA Dataset...")
    try:
        dataset = download_qa_dataset()
        logger.info("✅ Dataset downloaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Dataset download error: {e}")


def handle_experiment_selection(experiment_choice):
    """Handle the experiment selection from the experiments menu."""
    global vector_db, embedding_model, tokenizer, model, device, storing_method, source_path, distance_metric
    if experiment_choice == '1':
        run_retrieval_base_algorithm_experiment(vector_db=vector_db, embedding_model=embedding_model)
    elif experiment_choice == '2':
        run_evaluation_mode(vector_db, embedding_model, tokenizer, model, device)
    elif experiment_choice == '3':
        run_development_test(vector_db, embedding_model, tokenizer, model, device)
    elif experiment_choice == '4':
        run_analysis()
    elif experiment_choice == '5':
        show_system_info(vector_db, model, device)
    elif experiment_choice == '6':
        download_dataset()
    elif experiment_choice == '7':
        show_exit_message()
    elif experiment_choice == '8':
        if confirm_reset_vector_db():
            vector_db, embedding_model, storing_method, source_path, distance_metric = setup_vector_database()
            show_vector_db_success()


def main_loop():
    """Main application loop with clean menu system."""
    global vector_db, embedding_model, tokenizer, model, device, storing_method, source_path, distance_metric

    while True:
        choice = display_main_menu()

        if choice == '1':
            run_interactive_mode(vector_db, embedding_model, tokenizer, model, device)
        elif choice == '2':
            run_evaluation_mode(vector_db, embedding_model, tokenizer, model, device)
        elif choice == '3':
            run_development_test(vector_db, embedding_model, tokenizer, model, device)
        elif choice == '4':
            run_analysis()
        elif choice == '5':
            show_system_info(vector_db, model, device)
        elif choice == '6':
            download_dataset()
        elif choice == "7":
            experiment_choice = show_experiments_menu()
            handle_experiment_selection(experiment_choice)
        elif choice == "8":
            show_exit_message()
            break
        elif choice == "9":
            if confirm_reset_vector_db():
                vector_db, embedding_model, storing_method, source_path, distance_metric = setup_vector_database()
                show_vector_db_success()


def main():
    """Main function with enhanced MPS support and error handling."""
    # Additional MPS environment setup
    if torch.backends.mps.is_available():
        torch.backends.mps.allow_tf32 = False
        logger.info("MPS environment configured for compatibility")

    logger.info("Application started with enhanced MPS support.")

    global vector_db, embedding_model, tokenizer, model, device, storing_method, source_path, distance_metric
    try:
        # Unified command line argument handling
        if handle_command_line_args(sys.argv):
            return

        # Normal application flow
        device = display_startup_banner()
        tokenizer, model = startup_initialization()

        # Setup vector database
        vector_db, embedding_model, storing_method, source_path, distance_metric = setup_vector_database()

        # Run main application loop
        main_loop()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        show_exit_message()
    except Exception as e:
        logger.error(f"Application error: {e}")
        show_error_message(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always save performance data
        if PERFORMANCE_AVAILABLE:
            try:
                os.makedirs("results", exist_ok=True)
                performance_monitor.save_metrics("results/performance_metrics.json")
                logger.info("Performance metrics saved")
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {e}")

        if CACHE_AVAILABLE:
            cache_stats()

        # Ask user if they want to print the summary
        show_performance_summary_notice()

        try:
            if ask_yes_no("Would you like to print the performance summary now? (y/n): "):
                logger.info("\n📊 SESSION SUMMARY")
                logger.info("-" * 20)
                if PERFORMANCE_AVAILABLE:
                    performance_report()
                if CACHE_AVAILABLE:
                    cache_stats()
        except Exception as e:
            logger.warning(f"Failed to prompt user for printing summary: {e}")

        # MPS cleanup
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                logger.info("MPS cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")


def print_help():
    """Display comprehensive help information."""
    logger.info("🚀 RAG System - Command Line Usage (MPS Enhanced)")
    logger.info("=" * 60)
    logger.info("USAGE:")
    logger.info("  python main.py                 - Run interactive RAG system")
    logger.info("  python main.py --analyze       - Analyze logged results")
    logger.info("  python main.py --performance   - Show performance report")
    logger.info("  python main.py --cache-stats   - Show cache statistics")
    logger.info("  python main.py --clear-cache   - Clear all caches")
    logger.info("  python main.py --help          - Show this help")
    logger.info("\n🍎 MPS (Apple Silicon) Features:")
    logger.info("  • Automatic float32 conversion for compatibility")
    logger.info("  • BFloat16 issues automatically resolved")
    logger.info("  • Memory fallback to CPU when needed")
    logger.info("  • Conservative generation settings for stability")
    logger.info("  • Real-time device monitoring and diagnostics")
    logger.info("\n📚 Vector Database Support:")
    logger.info("  • ChromaDB - Advanced persistent vector database")
    logger.info("  • Simple Storage - LlamaIndex default storage")
    logger.info("  • Interactive setup wizard with validation")
    logger.info("  • Support for local, URL, and HuggingFace sources")
    logger.info("\n💬 Interactive Features:")
    logger.info("  • Natural language Q&A interface")
    logger.info("  • Real-time confidence scoring")
    logger.info("  • Device usage indicators")
    logger.info("  • Built-in help and statistics commands")
    logger.info("\n🔧 Troubleshooting:")
    logger.info("  • Use Development Test mode for diagnostics")
    logger.info("  • Check logs in logger.log for details")
    logger.info("  • Vector DB is optional for basic Q&A")


if __name__ == "__main__":
    main()
