# main.py
import asyncio
import os
import sys
import warnings

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from experiments.QACollection.cross_entropy_vs_retriever import run_entropy_correlation_experiment
from experiments.global_correlation_experiment import run_global_correlation_experiment_async
from experiments.llm_relevance_analysis import run_llm_relevance_experiment
from experiments.llm_score_vs_distance import run_llm_score_vs_distance_scatter_experiment, \
    run_retriever_rank_vs_distance_experiment
from experiments.retrieval_base_algorithm import run_retrieval_base_algorithm_experiment
from matrics.results_logger import ResultsLogger, plot_score_distribution
from scripts.qa_data_set_loader import download_qa_dataset
from utility.logger import logger
from utility.performance import (performance_report, performance_monitor, track_performance)
from utility.user_interface import display_main_menu, show_system_info, run_interactive_mode, run_evaluation_mode, \
    run_development_test, startup_initialization, setup_vector_database, display_startup_banner, ask_yes_no, \
    handle_command_line_args, show_exit_message, show_error_message, \
    confirm_reset_vector_db, show_vector_db_success, show_performance_summary_notice, show_experiments_menu, \
    cache_stats, run_noise_robustness_experiment, ask_selection, print_section_header
from vector_db.vector_db_interface import VectorDBInterface

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
model: torch.nn.Module | None = None
cross_encoder_model = None
vector_db: VectorDBInterface | None = None
embedding_model: HuggingFaceEmbedding | None = None
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
    global vector_db, embedding_model, tokenizer, model, cross_encoder_model, device, storing_method, source_path, distance_metric
    if experiment_choice == '1':
        run_retrieval_base_algorithm_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model,
            evaluator_model=model
        )

    elif experiment_choice == '2':
        run_noise_robustness_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model
        )

    elif experiment_choice == '3':
        run_development_test(vector_db, embedding_model, tokenizer, model, device)

    elif experiment_choice == '4':
        run_llm_score_vs_distance_scatter_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model,
            llm_tokenizer=tokenizer,
            llm_model=model,
            cross_encoder_model=cross_encoder_model,
        )

    elif experiment_choice == '5':
        run_retriever_rank_vs_distance_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model
        )
    elif experiment_choice == '6':
        run_llm_relevance_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model
        )
    elif experiment_choice == '7':
        run_entropy_correlation_experiment(
            vector_db=vector_db,
            embedding_model=embedding_model,
            scoring_model=cross_encoder_model
        )
    elif experiment_choice == '8':
        print_section_header("🌐 GLOBAL CORRELATION (BATCH API)")
        print(
            "\n"
            "[1] Pilot Mode (Test Run)\n"
            "  Description: Short run on a small number of queries (e.g., 5).\n"
            "  Goal: Validate DB wiring, JSONL format, and submission to Gemini Batch API.\n"
            "  Recommendation: Always run this before a full experiment.\n"
            "\n"
            "[2] Full Experiment Mode\n"
            "  Description: Run on the full query pool (e.g., 200) with batching of 20 docs/request.\n"
            "  Goal: Collect full correlation data. Requires higher quota and longer wait time.\n"
            "\n"
            "[3] Sync & Analyze (Harvester)\n"
            "  Description: Check last job status, then ingest the Batch output JSONL, update SQLite,\n"
            "  and generate plots/statistics.\n"
            "  Goal: Close the loop after the Batch job completes.\n"
            "\n"
            "[0] Back to Main Menu\n"
        )

        action = input("Select option (1/2/3/0) [default=1]: ").strip() or "1"
        if action == "0":
            return

        if action == "1":
            from experiments.global_correlation_experiment import run_global_correlation_pilot_batch_generator
            asyncio.run(run_global_correlation_pilot_batch_generator(
                vector_db=vector_db,
                embedding_model=embedding_model,
                num_queries=200,
                pilot_num_queries=5,
                k=100,
            ))
            return

        if action == "2":
            print(
                "\nWARNING: Full experiment will generate and submit a very large Batch job.\n"
                "This may take a long time and consume significant quota.\n"
            )
            if not ask_yes_no("Do you want to continue with FULL experiment?", default='False'):
                return
            from experiments.global_correlation_experiment import run_global_correlation_experiment_async
            asyncio.run(run_global_correlation_experiment_async(
                vector_db=vector_db,
                embedding_model=embedding_model,
                num_queries=200,
                k=100,
            ))
            return

        if action == "3":
            # Sync & Analyze: user provides paths; we optionally check last job status using job_id in experiment_meta.
            from experiments.global_correlation_experiment import check_batch_status
            from sync_batch_results import sync_batch_results
            import sqlite3 as _sqlite3

            # Best-effort default: latest run folder in results/global_exp
            default_db_path = ""
            try:
                base = os.path.join("results", "global_exp")
                if os.path.isdir(base):
                    dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
                    dirs = sorted(dirs, reverse=True)
                    if dirs:
                        guess = os.path.join(base, dirs[0], "experiment_results.db")
                        if os.path.exists(guess):
                            default_db_path = guess
            except Exception:
                default_db_path = ""

            db_prompt = "Path to experiment_results.db"
            if default_db_path:
                db_prompt += f" [default={default_db_path}]"
            db_prompt += ": "
            db_path = input(db_prompt).strip() or default_db_path
            if not db_path:
                print("No db_path provided. Aborting Sync & Analyze.")
                return

            output_prompt = (
                "\n"
                "Path to Batch OUTPUT JSONL file(s) (from Google)\n"
                "  (Examples: ~/Downloads/output.jsonl OR ~/Downloads/my_batch_folder/ OR ~/Downloads/*.jsonl)\n"
                "  Note: You can now provide a single file, a folder containing .jsonl files, or a glob pattern.\n"
                "  You must download these files from https://aistudio.google.com/app/batch\n"
                "  once the status shows 'SUCCEEDED'.\n"
                "> "
            )
            output_jsonl_path = input(output_prompt).strip()
            if not output_jsonl_path:
                print("No output_jsonl_path provided. Aborting Sync & Analyze.")
                return

            # Optional status check
            try:
                conn = _sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT value FROM experiment_meta WHERE key = ?", ("last_batch_job_id",))
                row = cur.fetchone()
                conn.close()
                job_id = str(row[0]) if row and row[0] else None
            except Exception:
                job_id = None

            if job_id:
                print(f"Checking last batch job status (job_id={job_id})...")
                try:
                    check_batch_status(job_id)
                except Exception as e:
                    print(f"Status check failed (continuing to sync anyway): {e}")

            sync_batch_results(db_path=db_path, output_paths_input=output_jsonl_path)
            return

        print("Invalid selection. Returning to menu.")
        return
    else:
        # Back to main menu
        return


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
        elif choice == "10":
            try:
                from experiments.global_correlation_experiment import _ollama_print_available_models_once
                _ollama_print_available_models_once()
            except Exception as e:
                logger.warning(f"Could not print Ollama models: {e}")


def main():
    """Main function with enhanced MPS support and error handling."""
    # Additional MPS environment setup
    if torch.backends.mps.is_available():
        torch.backends.mps.allow_tf32 = False
        logger.info("MPS environment configured for compatibility")

    logger.info("Application started with enhanced MPS support.")

    global vector_db, embedding_model, tokenizer, model, cross_encoder_model, device, storing_method, source_path, distance_metric
    try:
        # Unified command line argument handling
        if handle_command_line_args(sys.argv):
            return

        # Normal application flow
        device = display_startup_banner()
        tokenizer, model, cross_encoder_model = startup_initialization()

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
