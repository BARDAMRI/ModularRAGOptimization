# main.py
import asyncio
import os
import sys
import warnings

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from configurations import config
from experiments.QACollection.cross_entropy_vs_retriever import run_entropy_correlation_experiment
from experiments.global_correlation_experiment import ollama_print_available_models_once
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
    cache_stats, run_noise_robustness_experiment, print_section_header
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
            "[staged / 4] Staged Additive-Pool Mode (Recommended)\n"
            "  Description : Builds the scoring pool incrementally, STRIDE docs per stream per stage.\n"
            "  Mechanics   : Stage 1 seeds GT docs + first ranked+random slices; each subsequent\n"
            "                stage appends the next slice, scores all queries vs new docs, and\n"
            "                harvests results before proceeding. Manifest tracks cursors for safe\n"
            "                resume. Equivalent to full mode once all stages complete.\n"
            "  Goal        : Full correlation data with fine-grained resumability and no large\n"
            "                single-shot submissions.\n"
            "  Config keys : global_correlation.staging_stride, .staging_max_ranked_per_gt,\n"
            "                .staging_max_random_per_query, .queries_to_load\n"
            "\n"
            "[pilot / 1] Pilot Mode (Test Run)\n"
            "  Description : Short run on a small number of queries (e.g., 5).\n"
            "  Mechanics   : Generates a JSONL of Gemini Batch requests AND submits a tiny job\n"
            "                so you can validate plumbing end-to-end. Use Ollama provider for a\n"
            "                fully-local validation that does not need a Gemini API key.\n"
            "  Goal        : Validate DB wiring, JSONL format, and submission to Gemini Batch API.\n"
            "  Config keys : global_correlation.pilot_query_count, .queries_to_load (cap),\n"
            "                .retrieval_top_k, .ollama_max_docs_per_query, .gemini_docs_per_batch\n"
            "\n"
            "[full / 2] Full Experiment Mode\n"
            "  Description : Run on the full query pool (e.g., 200) with batching of 20 docs/request.\n"
            "  Mechanics   : Splits queries into chunks (default 25; override via run_config\n"
            "                gemini_pipeline_chunk_size) and submits up to N concurrent Batch jobs\n"
            "                from this run (default 2; override gemini_max_concurrent_batch_jobs).\n"
            "                Waits if account-wide queue hits Google's concurrent-batch limit (~100).\n"
            "                429 / network back-off, Files API cleanup (see gemini_aggressive_file_cleanup_before_upload).\n"
            "  Goal        : Collect full correlation data. Requires higher quota and longer wait time.\n"
            "  Config keys : .queries_to_load, .retrieval_top_k, .auto_confirm_full_run,\n"
            "                .gemini_pipeline_chunk_size, .gemini_max_concurrent_batch_jobs,\n"
            "                .gemini_aggressive_file_cleanup_before_upload\n"
            "\n"
            "[sync / 3] Sync & Analyze (Harvester)\n"
            "  Description : Check last job status, then ingest the Batch output JSONL, update SQLite,\n"
            "                and generate plots/statistics.\n"
            "  Goal        : Close the loop after the Batch job completes.\n"
            "  Config keys : global_correlation.sync_database_path, .sync_output_path\n"
            "\n"
            "[0] Back to Main Menu\n"
        )

        from utility.run_config import run_config as _rc

        # Normalize run_mode: config supplies "pilot"/"full"/"staged"/"sync", interactive "1"/"2"/"3"/"4"
        _RUN_MODE_MAP = {
            "1": "pilot",  "pilot":  "pilot",
            "2": "full",   "full":   "full",
            "3": "sync",   "sync":   "sync",
            "4": "staged", "staged": "staged",
            "0": "back",
        }

        # scoring_provider override — options validation already handled by _unwrap in run_config.py
        if _rc.enabled:
            _override = _rc.get_str("global_correlation.scoring_provider")
            if _override:
                try:
                    print(f"[config] global_correlation.scoring_provider = {_override!r} "
                          f"(was {config.CORRELATION_LLM_PROVIDER!r})")
                    config.CORRELATION_LLM_PROVIDER = _override
                except Exception as _e:
                    logger.warning(f"Could not override CORRELATION_LLM_PROVIDER: {_e}")

        _prov = str(config.CORRELATION_LLM_PROVIDER).strip().lower()
        if _prov == "ollama":
            ollama_print_available_models_once()
        elif _prov in ("nvidia_ih", "nvidia", "inference_hub"):
            from configurations.config import NVIDIA_IH_MODEL, NVIDIA_IH_API_KEY
            if not (NVIDIA_IH_API_KEY or "").strip():
                print("[NVIDIA_IH] Warning: IH_API_KEY / NVIDIA_IH_API_KEY is not set.")
            print(f"[NVIDIA_IH] Model: {NVIDIA_IH_MODEL}")

        _cfg_mode = None
        if _rc.enabled:
            _cfg_mode = _rc.get("global_correlation.run_mode") or _rc.get("global_correlation.action")
        if _cfg_mode:
            _raw_mode = str(_cfg_mode).strip().lower()
            run_mode = _RUN_MODE_MAP.get(_raw_mode, _raw_mode)
        else:
            _raw_mode = input("Select option (staged/pilot/full/sync or 4/1/2/3/0) [default=4]: ").strip() or "4"
            run_mode = _RUN_MODE_MAP.get(_raw_mode.lower(), _raw_mode.lower())

        if run_mode == "back":
            return

        if run_mode == "staged":
            from experiments.global_correlation_experiment import run_global_correlation_staged_async
            _num_queries = _rc.get_int("global_correlation.queries_to_load", 200) if _rc.enabled else 200
            _timeout_min = _rc.get_int("global_correlation.auto_harvest_per_job_timeout_min", 0) if _rc.enabled else 0
            _per_job_timeout_s = (_timeout_min * 60) if _timeout_min and _timeout_min > 0 else None
            _staged_kw: dict = {}
            if _rc.enabled:
                if _rc.has("global_correlation.staging_stride"):
                    _staged_kw["stride"] = _rc.get_int("global_correlation.staging_stride",
                                                        int(config.STAGING_STRIDE))
                if _rc.has("global_correlation.staging_max_ranked_per_gt"):
                    _staged_kw["max_ranked"] = _rc.get_int("global_correlation.staging_max_ranked_per_gt",
                                                            int(config.STAGING_MAX_RANKED_PER_GT))
                if _rc.has("global_correlation.staging_max_ranked_per_query"):
                    _staged_kw["max_ranked_query"] = _rc.get_int("global_correlation.staging_max_ranked_per_query",
                                                                   int(config.STAGING_MAX_RANKED_PER_QUERY))
                if _rc.has("global_correlation.staging_max_random_per_query"):
                    _staged_kw["max_random"] = _rc.get_int("global_correlation.staging_max_random_per_query",
                                                            int(config.STAGING_MAX_RANDOM_PER_QUERY))
            asyncio.run(run_global_correlation_staged_async(
                vector_db=vector_db,
                embedding_model=embedding_model,
                num_queries=_num_queries,
                per_job_timeout_s=_per_job_timeout_s,
                **_staged_kw,
            ))
            return

        if run_mode == "pilot":
            from experiments.global_correlation_experiment import run_global_correlation_pilot_batch_generator
            _num_queries = _rc.get_int("global_correlation.queries_to_load", 200) if _rc.enabled else 200
            _pilot_n = _rc.get_int("global_correlation.pilot_query_count", 5) if _rc.enabled else 5
            _k = _rc.get_int("global_correlation.retrieval_top_k", 100) if _rc.enabled else 100
            _pilot_batch = _rc.get_int("global_correlation.gemini_docs_per_batch", 20) if _rc.enabled else 20
            _timeout_min = _rc.get_int("global_correlation.auto_harvest_per_job_timeout_min", 0) if _rc.enabled else 0
            _per_job_timeout_s = (_timeout_min * 60) if _timeout_min and _timeout_min > 0 else None
            # Ollama pilot: optionally override CORRELATION_PILOT_MAX_DOCS_PER_QUERY from run_config
            if _rc.enabled and _rc.has("global_correlation.ollama_max_docs_per_query"):
                try:
                    config.CORRELATION_PILOT_MAX_DOCS_PER_QUERY = _rc.get_int(
                        "global_correlation.ollama_max_docs_per_query",
                        int(config.CORRELATION_PILOT_MAX_DOCS_PER_QUERY),
                    )
                    print(f"[config] global_correlation.ollama_max_docs_per_query = "
                          f"{config.CORRELATION_PILOT_MAX_DOCS_PER_QUERY}")
                except Exception as _e:
                    logger.warning(f"Could not override CORRELATION_PILOT_MAX_DOCS_PER_QUERY: {_e}")
            asyncio.run(run_global_correlation_pilot_batch_generator(
                vector_db=vector_db,
                embedding_model=embedding_model,
                num_queries=_num_queries,
                pilot_num_queries=_pilot_n,
                k=_k,
                pilot_batch_size=_pilot_batch,
                per_job_timeout_s=_per_job_timeout_s,
            ))
            return

        if run_mode == "full":
            print(
                "\nWARNING: Full experiment will generate and submit a very large Batch job.\n"
                "This may take a long time and consume significant quota.\n"
            )
            if _rc.enabled and _rc.get_bool("global_correlation.auto_confirm_full_run") is not None:
                confirmed = _rc.get_bool("global_correlation.auto_confirm_full_run")
            else:
                confirmed = ask_yes_no("Do you want to continue with FULL experiment?", default='False')
            if not confirmed:
                return
            from experiments.global_correlation_experiment import (
                MAX_CONCURRENT_JOBS,
                PIPELINE_CHUNK_SIZE,
                run_global_correlation_experiment_async,
            )
            _num_queries = _rc.get_int("global_correlation.queries_to_load", 200) if _rc.enabled else 200
            _k = _rc.get_int("global_correlation.retrieval_top_k", 100) if _rc.enabled else 100
            _timeout_min = _rc.get_int("global_correlation.auto_harvest_per_job_timeout_min", 0) if _rc.enabled else 0
            _per_job_timeout_s = (_timeout_min * 60) if _timeout_min and _timeout_min > 0 else None
            _gemini_kw: dict = {}
            if _rc.enabled:
                if _rc.has("global_correlation.gemini_pipeline_chunk_size"):
                    _gemini_kw["pipeline_chunk_size"] = _rc.get_int(
                        "global_correlation.gemini_pipeline_chunk_size",
                        PIPELINE_CHUNK_SIZE,
                    )
                if _rc.has("global_correlation.gemini_max_concurrent_batch_jobs"):
                    _gemini_kw["max_concurrent_batch_jobs"] = _rc.get_int(
                        "global_correlation.gemini_max_concurrent_batch_jobs",
                        MAX_CONCURRENT_JOBS,
                    )
                if _rc.has("global_correlation.gemini_aggressive_file_cleanup_before_upload"):
                    _gemini_kw["gemini_aggressive_file_cleanup_before_upload"] = bool(
                        _rc.get_bool(
                            "global_correlation.gemini_aggressive_file_cleanup_before_upload",
                            default=False,
                        )
                    )
            if _num_queries < 50:
                print(f"\n⚠️  WARNING: global_correlation.queries_to_load={_num_queries} is very small for a "
                      f"full experiment.\n    This is likely a pilot value left in run_config.json. "
                      f"Set queries_to_load=200 for a meaningful full run.\n")
            asyncio.run(run_global_correlation_experiment_async(
                vector_db=vector_db,
                embedding_model=embedding_model,
                num_queries=_num_queries,
                k=_k,
                per_job_timeout_s=_per_job_timeout_s,
                **_gemini_kw,
            ))
            return

        if run_mode == "sync":
            # Ingest Batch output JSONL, update SQLite, regenerate plots
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

            if _rc.enabled and _rc.get("global_correlation.sync_database_path"):
                db_path = str(_rc.get("global_correlation.sync_database_path"))
                print(f"[config] global_correlation.sync_database_path = {db_path!r}")
            else:
                db_prompt = "Path to experiment_results.db"
                if default_db_path:
                    db_prompt += f" [default={default_db_path}]"
                db_prompt += ": "
                db_path = input(db_prompt).strip() or default_db_path
            if not db_path:
                print("No db_path provided. Aborting Sync & Analyze.")
                return

            if _rc.enabled and _rc.get("global_correlation.sync_output_path"):
                output_jsonl_path = str(_rc.get("global_correlation.sync_output_path"))
                print(f"[config] global_correlation.sync_output_path = {output_jsonl_path!r}")
            else:
                output_prompt = (
                    "\n"
                    "Path to Batch OUTPUT JSONL file(s) (from Google)\n"
                    "  (Examples: ~/Downloads/output.jsonl OR ~/Downloads/my_batch_folder/ OR ~/Downloads/*.jsonl)\n"
                    "  Note: You can provide a single file, a folder containing .jsonl files, or a glob pattern.\n"
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

        if run_mode == "retry":
            # Re-submit ONLY the batch requests that ended with status missing/failed/pending.
            # Keeps the same run_dir and reuses the existing JSONL lines so the experiment
            # continues seamlessly. Auto-harvest takes over after submission.
            from experiments.global_correlation_experiment import retry_missing_batches
            _timeout_min = _rc.get_int("global_correlation.auto_harvest_per_job_timeout_min", 0) if _rc.enabled else 0
            _per_job_timeout_s = (_timeout_min * 60) if _timeout_min and _timeout_min > 0 else None

            # Locate run db
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

            if _rc.enabled and _rc.get("global_correlation.sync_database_path"):
                db_path = str(_rc.get("global_correlation.sync_database_path"))
                print(f"[config] global_correlation.sync_database_path = {db_path!r}")
            else:
                db_prompt = "Path to experiment_results.db (the run you want to retry)"
                if default_db_path:
                    db_prompt += f" [default={default_db_path}]"
                db_prompt += ": "
                db_path = input(db_prompt).strip() or default_db_path
            if not db_path or not os.path.exists(db_path):
                print(f"No valid db_path provided ({db_path!r}). Aborting Retry.")
                return

            retry_missing_batches(
                db_path=db_path,
                per_job_timeout_s=_per_job_timeout_s,
            )
            return

        print("Invalid selection. Returning to menu.")
        return
    else:
        # Back to main menu
        return


def main_loop():
    """Main application loop with clean menu system."""
    global vector_db, embedding_model, tokenizer, model, device, storing_method, source_path, distance_metric

    # When the user is driving the app from run_config.json we run a single
    # action and exit, otherwise the loop would re-trigger the same menu choice
    # forever (e.g. would resubmit a batch pilot job on every iteration).
    from utility.run_config import run_config as _rc_loop
    single_shot = bool(_rc_loop.enabled and _rc_loop.get_str("main_menu_choice"))

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
                ollama_print_available_models_once()
            except Exception as e:
                logger.warning(f"Could not print Ollama models: {e}")

        if single_shot:
            logger.info(
                "[run_config] single-shot mode: exiting main loop after one action. "
                "Set use_config_file=false (or clear main_menu_choice) to keep the menu open."
            )
            break


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
            from utility.run_config import run_config as _rc
            print_summary = None
            if _rc.enabled:
                cfg_val = _rc.get_bool("performance_summary.print_summary")
                if cfg_val is not None:
                    print_summary = cfg_val
                    print(f"[config] performance_summary.print_summary = {cfg_val!r}")
            if print_summary is None:
                print_summary = ask_yes_no("Would you like to print the performance summary now? (y/n): ")
            if print_summary:
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
    os._exit(0)
