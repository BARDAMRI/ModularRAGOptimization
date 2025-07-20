# main.py
import os
import sys
import warnings
import torch
from matrics.results_logger import ResultsLogger, plot_score_distribution
from scripts.qa_data_set_loader import download_qa_dataset
from utility.logger import logger
from utility.user_interface import display_main_menu, show_system_info, run_interactive_mode, run_evaluation_mode, \
    run_development_test, startup_initialization, setup_vector_database, display_startup_banner, ask_yes_no, \
    run_noise_robustness_experiment

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set MPS environment variables before importing PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.backends.mps.is_available():
    torch.backends.mps.allow_tf32 = False
    print("🔧 MPS configured for compatibility")

# Suppress FutureWarning specifically from huggingface_hub
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0.*",
    category=FutureWarning,
    module='huggingface_hub'
)

# Import performance monitoring with fallbacks
try:
    from utility.performance import (
        monitor_performance, performance_report, performance_monitor, track_performance
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
    print("\n📈 RESULTS ANALYSIS")
    print("=" * 25)

    logger.info("Running analysis on logged results...")
    try:
        logger_instance = ResultsLogger()
        logger_instance.summarize_scores()
        plot_score_distribution()
        print("✅ Analysis completed")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis error: {e}")


def download_dataset():
    """Download the QA dataset."""

    print("\n📥 Downloading QA Dataset...")
    try:
        dataset = download_qa_dataset()
        print("✅ Dataset downloaded successfully.")
        return dataset
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        logger.error(f"Dataset download error: {e}")


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
            run_noise_robustness_experiment(vector_db, embedding_model)
        elif choice == "8":
            print("👋 Goodbye!")
            break
        elif choice == "9":
            print("\n🔄 Resetting vector database and embedding model...\n")
            vector_db, embedding_model, storing_method, source_path, distance_metric = setup_vector_database()
            print("✅ Vector DB and embedding model reloaded.")


def main():
    """Main function with enhanced MPS support and error handling."""
    # Additional MPS environment setup
    if torch.backends.mps.is_available():
        torch.backends.mps.allow_tf32 = False
        logger.info("MPS environment configured for compatibility")

    logger.info("Application started with enhanced MPS support.")

    global vector_db, embedding_model, tokenizer, model, device, storing_method, source_path, distance_metric
    try:
        # Handle special command line arguments
        if "--help" in sys.argv:
            print_help()
            return

        if "--clear-cache" in sys.argv:
            if CACHE_AVAILABLE:
                clear_all_caches()
                print("✅ Cache cleared")
            return

        if "--analyze" in sys.argv:
            tokenizer, model = startup_initialization()
            run_analysis()
            return

        if "--performance" in sys.argv:
            if PERFORMANCE_AVAILABLE:
                performance_report()
            return

        if "--cache-stats" in sys.argv:
            if CACHE_AVAILABLE:
                cache_stats()
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
        print("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
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
        print("\n🧾 Performance summary saved to: results/performance_metrics.json")
        print("📊 You can view it later or print it now.")

        try:
            if ask_yes_no("Would you like to print the performance summary now? (y/n): "):
                print("\n📊 SESSION SUMMARY")
                print("-" * 20)
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
    print("🚀 RAG System - Command Line Usage (MPS Enhanced)")
    print("=" * 60)
    print("USAGE:")
    print("  python main.py                 - Run interactive RAG system")
    print("  python main.py --analyze       - Analyze logged results")
    print("  python main.py --performance   - Show performance report")
    print("  python main.py --cache-stats   - Show cache statistics")
    print("  python main.py --clear-cache   - Clear all caches")
    print("  python main.py --help          - Show this help")
    print("\n🍎 MPS (Apple Silicon) Features:")
    print("  • Automatic float32 conversion for compatibility")
    print("  • BFloat16 issues automatically resolved")
    print("  • Memory fallback to CPU when needed")
    print("  • Conservative generation settings for stability")
    print("  • Real-time device monitoring and diagnostics")
    print("\n📚 Vector Database Support:")
    print("  • ChromaDB - Advanced persistent vector database")
    print("  • Simple Storage - LlamaIndex default storage")
    print("  • Interactive setup wizard with validation")
    print("  • Support for local, URL, and HuggingFace sources")
    print("\n💬 Interactive Features:")
    print("  • Natural language Q&A interface")
    print("  • Real-time confidence scoring")
    print("  • Device usage indicators")
    print("  • Built-in help and statistics commands")
    print("\n🔧 Troubleshooting:")
    print("  • Use Development Test mode for diagnostics")
    print("  • Check logs in logger.log for details")
    print("  • Vector DB is optional for basic Q&A")


if __name__ == "__main__":
    main()
