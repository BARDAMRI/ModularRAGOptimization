# main.py - Enhanced MPS Support Version
import os
import sys
import warnings

from utility.device_utils import get_optimal_device

# Set MPS environment variables before importing PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

if torch.backends.mps.is_available():
    torch.backends.mps.allow_tf32 = False
    print("🔧 MPS configured for compatibility")

# Your existing imports
from modules.model_loader import load_model
from modules.indexer import load_vector_db
from configurations.config import INDEX_SOURCE_URL, NQ_SAMPLE_SIZE, MAX_NEW_TOKENS, TEMPERATURE, QUALITY_THRESHOLD, \
    MAX_RETRIES
import termios
from modules.query import process_query_with_context
from scripts.evaluator import enumerate_top_documents
import json
from matrics.results_logger import ResultsLogger, plot_score_distribution
from utility.logger import logger

# Suppress FutureWarning specifically from huggingface_hub related to resume_download
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.",
    category=FutureWarning,
    module='huggingface_hub'  # Specify the module to be more precise
)

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
print("🚀 Loading model with MPS support...")
with monitor_performance("startup_model_loading"):
    tokenizer, model = load_model()
    print(f"✅ Model loaded on device: {next(model.parameters()).device}")


@track_performance("gpu_profiling")
def profile_gpu():
    """
    Enhanced GPU profiling with MPS support.
    """
    device = get_optimal_device()

    if device.type == "mps":
        print("\nMPS (Apple Silicon) GPU detected:")
        print("  - Memory monitoring is limited on MPS")
        print("  - Using float32 precision for compatibility")
        # Try to get some basic info
        try:
            print(f"  - Model device: {next(model.parameters()).device}")
            print(f"  - Model dtype: {next(model.parameters()).dtype}")

            # Check for any remaining bfloat16 parameters
            bfloat16_count = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
            if bfloat16_count > 0:
                print(f"  ⚠️  Warning: {bfloat16_count} bfloat16 parameters detected!")
            else:
                print("  ✅ All parameters are MPS-compatible")

        except Exception as e:
            print(f"  - Could not get model info: {e}")
    elif device.type == "cuda":
        print("\nCUDA GPU Utilization:")
        print(torch.cuda.memory_summary(device="cuda"))
    else:
        print("\nNo GPU detected. Using CPU.")


def check_device():
    """
    Enhanced device checking with MPS-specific information.
    """
    device = get_optimal_device()

    if device.type == "mps":
        print("MPS backend is available. Using Apple Silicon GPU for acceleration.")
        print("  - Configured for float32 precision")
        print("  - BFloat16 compatibility issues resolved")
        print("  - Conservative generation settings enabled")
    elif device.type == "cuda":
        print("CUDA backend is available. Using CUDA for acceleration.")
    else:
        print("No GPU detected. Using CPU.")

    return device


def flush_input():
    """
    Flushes accidental keyboard input from the terminal buffer before reading input.
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception as e:
        # This error is normal in some environments (like PyCharm debugger)
        pass  # Don't print the error as it's often harmless


@track_performance("complete_query_evaluation")
def run_query_evaluation():
    """Enhanced query evaluation with MPS safety."""
    logger.info("Starting query evaluation...")
    profile_gpu()

    with monitor_performance("vector_db_loading"):
        logger.info("Loading external Vector DB...")
        try:
            vector_db, embedding_model = load_vector_db(logger=logger, source="url", source_path=INDEX_SOURCE_URL)
            print("✅ Vector DB loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector DB: {e}")
            vector_db, embedding_model = None, None
            logger.info("Continuing without vector DB...")
            print("⚠️  Vector DB not available - running in simple Q&A mode")

    run_mode = input("\nRun in enumeration mode? (y/n): ").strip().lower()

    if run_mode == "y":
        if vector_db is None:
            print("❌ Enumeration mode requires Vector DB. Please set up vector database first.")
            return

        with monitor_performance("enumeration_mode_execution"):
            from scripts.evaluator import hill_climb_documents

            mode_choice = input("\nSelect mode: (e)numeration / (h)ill climbing: ").strip().lower()
            results_logger = ResultsLogger(top_k=5, mode="hill" if mode_choice == "h" else "enum")
            nq_file_path = "data/user_query_datasets/natural-questions-master/nq_open/NQ-open.dev.jsonl"

            if not os.path.exists(nq_file_path):
                logger.error(f"NQ file not found at: {nq_file_path}")
                print(f"❌ Dataset file not found: {nq_file_path}")
                return

            with open(nq_file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= NQ_SAMPLE_SIZE:
                        break
                    data = json.loads(line)
                    query = data.get("question")
                    logger.info(f"Running for NQ Query #{i + 1}: {query}")
                    print(f"🔍 Processing query {i + 1}/{NQ_SAMPLE_SIZE}: {query[:50]}...")

                    with monitor_performance(f"query_processing_{i + 1}"):
                        if mode_choice == "h":
                            result = hill_climb_documents(i=i, num=NQ_SAMPLE_SIZE, query=query, index=vector_db,
                                                          llm_model=model, tokenizer=tokenizer,
                                                          embedding_model=embedding_model, top_k=5,
                                                          max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                                          quality_threshold=QUALITY_THRESHOLD, max_retries=MAX_RETRIES)
                            results_logger.log(result)
                        elif mode_choice == "e":
                            result = enumerate_top_documents(i=i, num=NQ_SAMPLE_SIZE, query=query, index=vector_db,
                                                             embedding_model=embedding_model,
                                                             top_k=5, convert_to_vector=False)
                            results_logger.log(result)
        return
    else:
        logger.info("Entering interactive query mode...")
        device = check_device()

        print("\n💬 Interactive Q&A Mode")
        print("=" * 30)
        if vector_db is None:
            print("📝 Running in simple Q&A mode (no context retrieval)")
        else:
            print("📚 Running with context retrieval enabled")
        print("💡 Type 'exit' to quit\n")

        while True:
            flush_input()
            user_prompt = input("🤔 Your question: \n\n")
            if user_prompt.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting application.")
                print("👋 Goodbye!")
                break

            if not user_prompt.strip():
                print("Please enter a question.")
                continue

            print("🧠 Thinking...")
            with monitor_performance("interactive_query"):
                try:
                    result = process_query_with_context(
                        user_prompt, model, tokenizer, device,
                        vector_db, embedding_model,
                        max_retries=3,
                        quality_threshold=0.5
                    )

                    if result["error"]:
                        logger.error(f"Error: {result['error']}")
                        print(f"❌ Sorry, there was an error: {result['error']}")
                    else:
                        logger.info(f"Question: {result['question']}, Answer: {result['answer'].strip()}")
                        print(f"\n🤖 Answer: {result['answer'].strip()}")
                        if 'score' in result:
                            print(f"📊 Confidence: {result['score']:.2f}")
                        print(f"🖥️  Device: {result.get('device_used', 'unknown')}")
                        print("-" * 50)

                except Exception as e:
                    logger.error(f"Unexpected error during query processing: {e}")
                    print(f"❌ Unexpected error: {e}")


@track_performance("results_analysis")
def run_analysis():
    """
    Runs analysis on logged results, summarizing scores and plotting score distributions.
    """
    logger.info("Running analysis on logged results...")
    logger_instance = ResultsLogger()
    logger_instance.summarize_scores()  # Print average, min, max
    plot_score_distribution()  # Show histogram of score distribution


@track_performance("development_test")
def run_development_test():
    """Enhanced development test with MPS compatibility."""
    print("Development Mode - MPS Enhanced")
    print("=" * 40)

    # System info
    device = check_device()
    profile_gpu()

    # Test MPS compatibility
    print("\n🧪 Testing MPS compatibility...")
    try:
        from modules.model_loader import test_mps_compatibility
        mps_test_result = test_mps_compatibility()
        print(f"MPS Test: {'✅ PASSED' if mps_test_result else '❌ FAILED'}")
    except Exception as e:
        print(f"MPS Test: ❌ FAILED - {e}")

    # Quick model test
    print("\n🤖 Testing model capabilities...")
    from modules.model_loader import get_model_capabilities
    capabilities = get_model_capabilities(model)
    for key, value in capabilities.items():
        status = "✅" if value else "⚠️" if key == 'is_causal_lm' else "ℹ️"
        print(f"  {status} {key}: {value}")

    # Vector DB test (optional)
    print("\n📚 Testing vector database...")
    try:
        with monitor_performance("dev_vector_db_test"):
            vector_db, embedding_model = load_vector_db(logger=logger, source="url", source_path=INDEX_SOURCE_URL)
            print(f"✅ Vector DB loaded: {type(vector_db).__name__}")
    except Exception as e:
        print(f"⚠️  Vector DB test failed: {e}")
        vector_db, embedding_model = None, None

    # Query test
    print("\n💬 Testing query processing...")
    test_query = "What is artificial intelligence?"
    try:
        with monitor_performance("dev_query_test"):
            result = process_query_with_context(test_query, model, tokenizer, device, vector_db, embedding_model)

        print(f"✅ Test query result: {result['answer'][:100]}...")
        if 'score' in result:
            print(f"📊 Score: {result['score']}")
        print(f"🖥️  Device used: {result.get('device_used', 'unknown')}")
    except Exception as e:
        print(f"❌ Query test failed: {e}")

    print("\n🎉 Development test completed")


def main():
    """Main function with enhanced MPS support and error handling."""
    # Additional MPS environment setup
    if torch.backends.mps.is_available():
        torch.backends.mps.allow_tf32 = False
        logger.info("MPS environment configured for compatibility")

    logger.info("Application started with enhanced MPS support.")

    try:
        # Handle command line arguments
        if "--clear-cache" in sys.argv:
            if CACHE_AVAILABLE:
                clear_all_caches()
                print("✅ Cache cleared")
            else:
                print("❌ Cache functionality not available")

        if "--analyze" in sys.argv:
            run_analysis()
        elif "--performance" in sys.argv:
            if PERFORMANCE_AVAILABLE:
                performance_report()
            else:
                print("❌ Performance monitoring not available")
        elif "--cache-stats" in sys.argv:
            if CACHE_AVAILABLE:
                cache_stats()
            else:
                print("❌ Cache monitoring not available")
        elif "--dev" in sys.argv:
            run_development_test()
        else:
            run_query_evaluation()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Print final reports
        if PERFORMANCE_AVAILABLE:
            print("\n📊 Final Performance Report:")
            performance_report()

            # Save metrics for later analysis
            try:
                os.makedirs("results", exist_ok=True)
                performance_monitor.save_metrics("results/performance_metrics.json")
                logger.info("Performance metrics saved")
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {e}")

        if CACHE_AVAILABLE:
            print("\n💾 Final Cache Statistics:")
            cache_stats()

        # MPS cleanup
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                logger.info("MPS cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")


def print_help():
    print("🚀 RAG System - Command Line Usage (MPS Enhanced)")
    print("=" * 60)
    print("USAGE:")
    print("  python main.py                 - Run interactive RAG system")
    print("  python main.py --analyze       - Analyze logged results")
    print("  python main.py --performance   - Show performance report")
    print("  python main.py --cache-stats   - Show cache statistics")
    print("  python main.py --clear-cache   - Clear all caches")
    print("  python main.py --dev           - Run development test")
    print("  python main.py --help          - Show this help")
    print("\n🍎 MPS (Apple Silicon) Features:")
    print("  • Automatic float32 conversion for compatibility")
    print("  • BFloat16 issues automatically resolved")
    print("  • Memory fallback to CPU when needed")
    print("  • Conservative generation settings for stability")
    print("  • Real-time device monitoring and diagnostics")
    print("\n💬 Interactive Mode:")
    print("  • Type your questions naturally")
    print("  • Use 'exit', 'quit', or 'q' to quit")
    print("  • Watch for device indicators: 🖥️ MPS/CUDA/CPU")
    print("  • Confidence scores show answer quality")
    print("\n🔧 Troubleshooting:")
    print("  • Use --dev mode to test your setup")
    print("  • Check logs in logger.log for details")
    print("  • Vector DB is optional for basic Q&A")


if __name__ == "__main__":
    if "--help" in sys.argv:
        print_help()
    else:
        main()
