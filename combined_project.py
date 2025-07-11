# === File: combine.py ===
import os

root_dir = "/Users/bardamri/PycharmProjects/ModularRAGOptimization"
output_file_path = "combined_project.py"

excluded_dirs = {"__pycache__", ".venv", "env", ".git", ".idea", "build", "dist", "tests", "user_query_datasets"}

python_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

    for filename in filenames:
        if filename.endswith(".py") and filename != os.path.basename(output_file_path):
            full_path = os.path.join(dirpath, filename)
            python_files.append(full_path)

with open(output_file_path, "w", encoding="utf-8") as outfile:
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, root_dir)
        outfile.write(f"# === File: {rel_path} ===\n")
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
        except UnicodeDecodeError:
            print(f" file  {file_path} wasn't red with -UTF-8. moving to latin-1")
            with open(file_path, "r", encoding="latin-1") as infile:
                content = infile.read()
        outfile.write(content)
        outfile.write("\n\n")

print(f" combined file was created: {output_file_path}")


# === File: docsDownloader.py ===
import subprocess
import os


def download_llamaindex_docs(url, destination_folder):
    """
    Downloads the LlamaIndex documentation locally for offline use using httrack instead of wget.
    """
    os.makedirs(destination_folder, exist_ok=True)
    command = [
        "httrack",
        url,
        "-O", destination_folder,
        "--mirror",
        "--keep-alive",
        "--quiet"
    ]

    print(f"Downloading LlamaIndex documentation into '{destination_folder}'...")
    subprocess.run(command)
    print(f"Download complete! You can browse the docs offline from '{destination_folder}'.")


URL = input('insert URL: \t')
name = input('insert Name: \t')
if name and len(name) > 0 and URL and len(URL) > 0:
    download_llamaindex_docs(url=URL, destination_folder=name)


# === File: main.py ===
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
from configurations.config import INDEX_SOURCE_URL, NQ_SAMPLE_SIZE, TEMPERATURE
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
            vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
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

        print("\n💬 Interactive Q&A Mode")
        print("=" * 30)
        if vector_db is None:
            print("📝 Running in simple Q&A mode (no context retrieval)")
        else:
            print("📚 Running with context retrieval enabled")
        print("💡 Type 'exit' to quit\n")

        while True:
            flush_input()
            user_prompt = input("🤔 Your question: ")
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
            vector_db, embedding_model = load_vector_db("url", INDEX_SOURCE_URL)
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


# === File: configurations/config.py ===
# config.py - Light QA-Ready Configuration for Low-RAM/No-GPU systems

# ==========================
# ✅ ACTIVE MODEL CONFIGURATION (for QA on CPU / M1 / 16GB RAM)
# ==========================

MODEL_PATH = "tiiuae/falcon-rw-1b"                     # 🐦 Falcon 1B - very lightweight, extremely fast, basic QA

# ==========================
# Optional lightweight models (uncomment to switch)
# ==========================
# MODEL_PATH = "microsoft/phi-2"  # 🧠 Phi-2 (2.7B) - small, high-quality, works well on CPU with low RAM
# MODEL_PATH = "EleutherAI/gpt-neo-1.3B"                 # 🤖 GPT-Neo 1.3B - simple, good compatibility, fair QA
# MODEL_PATH = "openchat/openchat-3.5-0106"              # 💬 OpenChat 3.5 (3.5B) - solid QA/dialogue, quantize for better speed

# ==========================
# Do NOT use these unless you have 24GB+ VRAM or offloading infra
# ==========================
# MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"       # ⚠️ Heavy - requires 14GB+ RAM/VRAM
# MODEL_PATH = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  # ⚠️ Heavy but high quality (7B)
# MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"            # ⚠️ 7B - accurate, not suitable for low RAM
# MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"     # ❌ 13B+, MoE - too large
# MODEL_PATH = "meta-llama/Llama-2-13b-chat-hf"           # ❌ 13B - very heavy

# ==========================
# Embedding model
# ==========================
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === DATA PATHS ===
DATA_PATH = "data/public_corpus/"
PUBLIC_CORPUS_DATASET = "wikitext"
PUBLIC_CORPUS_DIR = "data/public_corpus"

# === DEVICE CONFIGURATION ===
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# === LLM SETTINGS ===
LLM_MODEL_NAME = MODEL_PATH
LLAMA_MODEL_DIR = MODEL_PATH

# === OPTIMIZATION PARAMETERS ===
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 64
NQ_SAMPLE_SIZE = 5

# === DATA SOURCE ===
DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"

# === GPU OPTIMIZATION SETTINGS ===
FORCE_CPU = False  # Set to True to force CPU usage
OPTIMIZE_FOR_MPS = True
MAX_GPU_MEMORY_GB = 8
USE_MIXED_PRECISION = False  # Recommended: False for MPS

# === File: matrics/analyze_results.py ===
from results_logger import ResultsLogger, plot_score_distribution


def main():
    logger = ResultsLogger()

    print("📊 Summarizing scores from logged results...\n")
    logger.summarize_scores()

    print("\n📈 Displaying histogram of score distribution...\n")
    plot_score_distribution()


if __name__ == "__main__":
    main()


# === File: matrics/results_logger.py ===
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime


class ResultsLogger:
    def __init__(self, filepath=None, truncate_fields=None, max_length=500, top_k=None, mode=None):
        """
        Initialize the logger. If filepath is not given, generate one using timestamp, top_k, and mode.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["results"]

        if mode:
            filename_parts.append(mode)
        if top_k:
            filename_parts.append(f"top{top_k}")
        filename_parts.append(timestamp)

        auto_filename = "_".join(filename_parts) + ".jsonl"
        self.filepath = filepath or os.path.join("results", auto_filename)
        self.max_length = max_length
        self.truncate_fields = truncate_fields or {"query", "answer", "context"}

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def _truncate(self, value):
        if isinstance(value, str) and len(value) > self.max_length:
            return value[:self.max_length] + "...[truncated]"
        return value

    def log(self, record: dict):
        safe_record = {
            k: self._truncate(v) if k in self.truncate_fields else v
            for k, v in record.items()
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

    def load_results(self):
        """Load all logged results from the output file."""
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def summarize_scores(self):
        """Prints basic statistics (count, average, min, max) for similarity scores."""
        results = self.load_results()
        scores = [r["score"] for r in results if "score" in r and isinstance(r["score"], (int, float))]
        if not scores:
            print("No scores to summarize.")
            return

        print(f"\n📊 Results Summary:")
        print(f"Total entries with score: {len(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.4f}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")


# --- Histogram plotting function ---
def plot_score_distribution(filepath="results/outputs.jsonl"):
    """Plot a histogram of similarity scores from the results file."""
    if not os.path.exists(filepath):
        print(f"No results file found at {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        scores = [
            json.loads(line.strip()).get("score")
            for line in f if line.strip()
        ]

    scores = [s for s in scores if isinstance(s, (int, float))]

    if not scores:
        print("No valid scores found to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=10, color="skyblue", edgecolor="black")
    plt.title("Similarity Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === File: scripts/create_public_corpus.py ===
# create_public_corpus.py
import os
from datasets import load_dataset
from configurations.config import PUBLIC_CORPUS_DATASET, PUBLIC_CORPUS_DIR

# Load a small public corpus from the "wikitext" dataset
dataset = load_dataset("wikitext", PUBLIC_CORPUS_DATASET, split="train")

# Directory where we'll store the text files
data_dir = '../' + PUBLIC_CORPUS_DIR
os.makedirs(data_dir, exist_ok=True)

# Save each non-empty document to a separate text file
for i, example in enumerate(dataset):
    text = example["text"]
    if text.strip():  # only write non-empty documents
        with open(os.path.join(data_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    print(f">Saved {i + 1} documents to '{data_dir}'")


# === File: scripts/dir_tree_printer.py ===
# dir_tree_printer.py
import os


def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def print_tree_summary(startpath, max_files_per_dir=5, depth=0):
    if not os.path.isdir(startpath):
        return

    entries = list(os.scandir(startpath))
    files = sorted([entry.name for entry in entries if entry.is_file()])
    dirs = sorted([entry for entry in entries if entry.is_dir()], key=lambda d: d.name)

    indent = '    ' * depth
    total_size = sum(os.path.getsize(os.path.join(startpath, f)) for f in files)
    print(f"{indent}{os.path.basename(startpath)}/ ({len(files)} files, {human_readable_size(total_size)})")

    for f in files[:max_files_per_dir]:
        display_name = f if len(f) <= 30 else f[:30] + "..."
        print(f"{indent}    {display_name}")
    if len(files) > max_files_per_dir:
        print(f"{indent}    ... +{len(files) - max_files_per_dir} more files")

    for d in dirs:
        print_tree_summary(d.path, max_files_per_dir, depth + 1)


print_tree_summary('../', max_files_per_dir=20)


# === File: scripts/evaluator.py ===
# evaluator.py - Final polished version
from typing import Tuple, Union, List, Dict, Any, Optional
import torch
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM
)
from configurations.config import HF_MODEL_NAME, LLM_MODEL_NAME
from utility.logger import logger
from utility.similarity_calculator import calculate_similarity, calculate_similarities, SimilarityMethod
from utility.embedding_utils import get_text_embedding


def load_llm() -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Load the LLM model and tokenizer.
    Returns (model, tokenizer) to match process_query_with_context expectations.
    """
    logger.info("Loading LLM model and tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        logger.info(f"Loaded causal language model: {LLM_MODEL_NAME}")
    except ValueError as e:
        logger.warning(f"Failed to load as causal model: {e}")
        model = AutoModel.from_pretrained(LLM_MODEL_NAME)
        logger.warning(f"Loaded masked language model instead: {LLM_MODEL_NAME}")

    return model, tokenizer


def load_model() -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Alias for load_llm() to match main.py expectations."""
    return load_llm()


def load_embedding_model() -> HuggingFaceEmbedding:
    """Load using LlamaIndex's HuggingFaceEmbedding wrapper."""
    logger.info(f"Loading embedding model: {HF_MODEL_NAME}")
    embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
    return embedding_model


def run_llm_query(
        query: str,
        model: Union[PreTrainedModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer
) -> str:
    """Run a query using the LLM."""
    logger.info(f"Running query: {query[:100]}...")  # Truncate long queries in logs
    inputs = tokenizer(query, return_tensors="pt")

    if hasattr(model, "generate"):
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated result: {result[:100]}...")  # Truncate long results
    else:
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        result = str(embeddings)
        logger.info("Computed embeddings from masked model")

    return result


def enumerate_top_documents(
        i: int,
        num: int,
        query: str,
        index: Any,
        embedding_model: HuggingFaceEmbedding,  # Fixed: Added type annotation
        top_k: int = 5,
        convert_to_vector: bool = False,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> Dict[str, Any]:
    """
    Enumerate top documents using the new similarity calculator.

    Args:
        i: Current query vector_db
        num: Total number of queries
        query: Query string
        index: Vector database vector_db
        embedding_model: HuggingFace embedding model
        top_k: Number of top documents to retrieve
        convert_to_vector: Whether to convert query to vector
        similarity_method: Similarity calculation method

    Returns:
        Dict containing query results and top documents
    """
    logger.info(f"Enumerating top documents for query #{i + 1} of {num}: {query[:50]}... using {similarity_method}")

    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    if convert_to_vector:
        query_vector = get_text_embedding(query, embedding_model)
    else:
        query_vector = query

    results = retriever.retrieve(query_vector)

    # Enhanced scoring using configurable similarity methods
    enhanced_results = []
    if convert_to_vector and results:
        # Get embeddings for all retrieved documents
        doc_texts = [node_with_score.node.get_content() for node_with_score in results]
        doc_embeddings = np.array([get_text_embedding(text, embedding_model) for text in doc_texts])

        # Calculate similarities using the new system
        if len(doc_embeddings) > 0:
            similarities = calculate_similarities(query_vector, doc_embeddings, similarity_method)

            # Combine with original results - Fixed: Use different variable name to avoid confusion
            for rank, (node_with_score, custom_score) in enumerate(zip(results, similarities), 1):
                content = node_with_score.node.get_content()
                enhanced_results.append({
                    "rank": rank,
                    "original_score": getattr(node_with_score, "score", None),
                    "custom_score": float(custom_score),
                    "similarity_method": str(similarity_method),
                    "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
                })
    else:
        # Fallback to original scoring
        for rank, node_with_score in enumerate(results, start=1):
            content = node_with_score.node.get_content()
            enhanced_results.append({
                "rank": rank,
                "original_score": getattr(node_with_score, "score", None),
                "custom_score": None,
                "similarity_method": "llamaindex_default",
                "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
            })

    result = {
        "query": query,
        "similarity_method": str(similarity_method),
        "convert_to_vector": convert_to_vector,
        "top_documents": enhanced_results,
        "total_documents": len(enhanced_results)
    }
    logger.info(f"Top documents enumerated using {similarity_method}: {len(enhanced_results)} documents")
    return result


def hill_climb_documents(
        i: int,
        num: int,
        query: str,
        index: Any,
        llm_model: Union[PreTrainedModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer,
        embedding_model: HuggingFaceEmbedding,  # Fixed: Added type annotation
        top_k: int = 5,
        max_tokens: int = 100,
        convert_to_vector: bool = False,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> Dict[str, Any]:
    """
    Perform hill climbing with configurable similarity methods.

    Args:
        i: Current query vector_db
        num: Total number of queries
        query: Query string
        index: Vector database vector_db
        llm_model: Language model for generating answers
        tokenizer: Tokenizer for the language model
        embedding_model: HuggingFace embedding model
        top_k: Number of top documents to retrieve
        max_tokens: Maximum tokens for generated answers
        convert_to_vector: Whether to convert query to vector
        similarity_method: Similarity calculation method

    Returns:
        Dict containing query results and best answer
    """
    logger.info(f"Hill climbing for query #{i + 1} of {num}: {query[:50]}... using {similarity_method}")

    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    if convert_to_vector:
        query_vector = get_text_embedding(query, embedding_model)
    else:
        query_vector = query

    results = retriever.retrieve(query_vector)

    best_score = -1.0
    best_answer = None
    best_context = None
    best_method_info = {"method": str(similarity_method), "score": best_score}

    contexts = [node_with_score.node.get_content() for node_with_score in results]
    if not contexts:
        logger.warning("No contexts retrieved.")
        return {"query": query, "answer": None, "context": None, "method_info": best_method_info}

    # Generate answers for all contexts
    prompts = [f"Context:\n{context}\n\nQuestion: {query}\nAnswer:" for context in contexts]

    # Handle tokenization with proper truncation
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = llm_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        answers = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        return {"query": query, "answer": None, "context": None, "method_info": best_method_info}

    # Use modular similarity calculator for scoring
    if convert_to_vector:
        query_emb = get_text_embedding(query, embedding_model)

        for context, answer in zip(contexts, answers):
            if not answer.strip():
                continue

            try:
                answer_emb = get_text_embedding(answer, embedding_model)
                # Use the new similarity calculator
                score = calculate_similarity(query_emb, answer_emb, similarity_method)

                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_context = context
                    best_method_info = {
                        "method": str(similarity_method),
                        "score": float(best_score)
                    }
            except Exception as e:
                logger.warning(f"Error calculating similarity for answer: {e}")
                continue
    else:
        # Fallback for non-vector mode
        for context, answer in zip(contexts, answers):
            if not answer.strip():
                continue

            # Simple word overlap heuristic
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            score = len(query_words & answer_words) / max(len(query_words), 1)

            if score > best_score:
                best_score = score
                best_answer = answer
                best_context = context
                best_method_info = {
                    "method": "word_overlap",
                    "score": float(best_score)
                }

    logger.info(f"Best answer selected using {similarity_method} with score {best_score:.4f}")
    return {
        "query": query,
        "answer": best_answer,
        "context": best_context,
        "method_info": best_method_info,
        "total_contexts_evaluated": len(contexts)
    }


def compare_answers_with_embeddings(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> str:
    """
    Compare answers using the new similarity calculator system.

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        similarity_method: Similarity calculation method

    Returns:
        str: "Optimized", "Original", or "Tie"
    """
    logger.info(f"Comparing answers using {similarity_method} similarity")

    try:
        embedding_model = load_embedding_model()

        # Generate embeddings
        query_embedding = get_text_embedding(user_query, embedding_model)
        orig_embedding = get_text_embedding(original_answer, embedding_model)
        opt_embedding = get_text_embedding(optimized_answer, embedding_model)

        # Use modular similarity calculator
        sim_orig = calculate_similarity(query_embedding, orig_embedding, similarity_method)
        sim_opt = calculate_similarity(query_embedding, opt_embedding, similarity_method)

        logger.info(f"Similarity scores using {similarity_method} - Original: {sim_orig:.4f}, Optimized: {sim_opt:.4f}")

        # More robust tie detection
        score_diff = abs(sim_opt - sim_orig)
        if score_diff < 0.001:  # Very close scores
            return "Tie"
        elif score_diff < 0.01:  # Close scores - use relative difference
            relative_diff = score_diff / max(abs(sim_orig), abs(sim_opt), 1e-6)
            if relative_diff < 0.05:  # Less than 5% relative difference
                return "Tie"

        return "Optimized" if sim_opt > sim_orig else "Original"

    except Exception as e:
        logger.error(f"Error in answer comparison: {e}")
        return "Tie"


def multi_method_comparison(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        methods: Optional[List[Union[SimilarityMethod, str]]] = None
) -> Dict[str, Any]:
    """
    Compare answers using multiple similarity methods for robust evaluation.

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        methods: List of similarity methods to use

    Returns:
        Dict containing results from all methods with final consensus
    """
    if methods is None:
        methods = [SimilarityMethod.COSINE, SimilarityMethod.DOT_PRODUCT, SimilarityMethod.EUCLIDEAN]

    logger.info(f"Multi-method comparison using {len(methods)} similarity methods")

    results = {}
    votes = {"Original": 0, "Optimized": 0, "Tie": 0}
    scores = {"Original": [], "Optimized": [], "Tie": []}

    for method in methods:
        try:
            result = compare_answers_with_embeddings(user_query, original_answer, optimized_answer, method)
            results[str(method)] = result
            votes[result] += 1

            # Could add actual similarity scores here for weighted voting
            # scores[result].append(score)

        except Exception as e:
            logger.warning(f"Error with method {method}: {e}")
            results[str(method)] = "Tie"
            votes["Tie"] += 1

    # Determine consensus
    consensus = max(votes, key=votes.get)
    confidence = votes[consensus] / len(methods)

    final_result = {
        "individual_results": results,
        "votes": votes,
        "consensus": consensus,
        "confidence": confidence,
        "methods_used": [str(m) for m in methods],
        "total_methods": len(methods)
    }

    logger.info(f"Multi-method consensus: {consensus} (confidence: {confidence:.2f})")
    return final_result


def judge_with_llm(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        model: Optional[Union[PreTrainedModel, AutoModelForCausalLM]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None
) -> str:
    """
    Judge answers using a language model (LLM).

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        model: LLM model instance (optional)
        tokenizer: Tokenizer for the LLM (optional)
        device: Device to run the model on (optional)

    Returns:
        str: "Optimized", "Original", or "Tie"
    """
    logger.info("Judging answers with LLM.")

    if model is None or tokenizer is None:
        model, tokenizer = load_llm()

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if hasattr(model, 'device') and str(model.device) != device:
            model = model.to(device)

        prompt = f"""You are an AI judge evaluating query optimization. Compare the two answers below and choose the best one.

Query: {user_query}

Original Answer: {original_answer}

Optimized Answer: {optimized_answer}

Answer ONLY with exactly one word: "Optimized", "Original", or "Tie". Do not include any extra text.
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        if device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        logger.info(f"LLM judgment result: {answer}")

        # More robust answer parsing
        answer_lower = answer.lower()
        for option in ["optimized", "original", "tie"]:
            if option in answer_lower:
                return option.capitalize()

        # Fallback - look for partial matches
        if "optim" in answer_lower:
            return "Optimized"
        elif "origin" in answer_lower:
            return "Original"
        else:
            return "Tie"

    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return "Tie"


def advanced_sanity_check(
        user_query: str,
        optimized_user_query: str,
        vector_db: Any,
        vector_query: Optional[Any] = None,
        similarity_methods: Optional[List[Union[SimilarityMethod, str]]] = None
) -> Dict[str, Any]:
    """
    Enhanced sanity check using multiple similarity methods.

    Provides more robust evaluation by testing different similarity approaches.

    Args:
        user_query: Original user query
        optimized_user_query: Optimized version of the query
        vector_db: Vector database instance
        vector_query: Optional vector representation of the query
        similarity_methods: List of similarity methods to use

    Returns:
        Dict containing comprehensive evaluation results
    """
    if similarity_methods is None:
        similarity_methods = [SimilarityMethod.COSINE, SimilarityMethod.DOT_PRODUCT]

    print(f"\n> Running advanced sanity check with {len(similarity_methods)} similarity methods...")

    try:
        model, tokenizer = load_llm()

        # Get answers (simplified for this example)
        orig_answer = run_llm_query(user_query, model, tokenizer)
        opt_answer = run_llm_query(optimized_user_query, model, tokenizer)

        # Multi-method embedding comparison
        multi_method_result = multi_method_comparison(user_query, orig_answer, opt_answer, similarity_methods)

        # Traditional LLM judgment
        llm_judgment = judge_with_llm(user_query, orig_answer, opt_answer, model=model, tokenizer=tokenizer)

        # Display results
        print("\n🔍 **Advanced Sanity Check Results**:")
        print(f"📝 Original Query: {user_query}")
        print(f"📝 Optimized Query: {optimized_user_query}")
        print(f"💬 Original Answer: {orig_answer[:200]}...")
        print(f"💬 Optimized Answer: {opt_answer[:200]}...")
        print(f"📊 LLM Judgment: {llm_judgment}")
        print(
            f"📊 Multi-method Results: {multi_method_result['consensus']} (confidence: {multi_method_result['confidence']:.2f})")
        print(f"📊 Individual Method Results: {multi_method_result['individual_results']}")

        # Determine final decision with weighted voting
        embedding_confidence = multi_method_result['confidence']
        llm_confidence = 0.7  # Fixed weight for LLM judgment

        final_decision = multi_method_result['consensus']
        if llm_judgment != multi_method_result['consensus'] and llm_confidence > embedding_confidence:
            final_decision = llm_judgment
            decision_reason = "LLM judgment overrode multi-method consensus due to higher confidence"
        else:
            decision_reason = "Multi-method consensus accepted"

        print(f"⚖️ Final Decision: {final_decision} ({decision_reason})")

        return {
            "original_query": user_query,
            "optimized_query": optimized_user_query,
            "original_answer": orig_answer,
            "optimized_answer": opt_answer,
            "llm_judgment": llm_judgment,
            "multi_method_result": multi_method_result,
            "final_decision": final_decision,
            "decision_reason": decision_reason,
            "confidence_scores": {
                "embedding_confidence": embedding_confidence,
                "llm_confidence": llm_confidence
            },
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in advanced sanity check: {e}")
        return {
            "error": str(e),
            "success": False
        }

# === File: scripts/modelHuggingFaceDownload.py ===
# modelHuggingFaceDownload.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from configurations.config import MODEL_PATH
from configurations.config import LLAMA_MODEL_DIR

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")

# Save them inside your project
model.save_pretrained(LLAMA_MODEL_DIR)
tokenizer.save_pretrained(LLAMA_MODEL_DIR)

print("> Model downloaded successfully!")


# === File: vector_db/chroma_index.py ===
# modules/vector_db/chroma_index.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient


def build_chroma_vector_db(source: str, source_path: str, embedding_model) -> VectorStoreIndex:
    """
    Build a Chroma-based vector database vector_db.

    Args:
        source (str): 'local' or 'url'
        source_path (str): Path or identifier for the source.
        embedding_model: Embedding model to use.

    Returns:
        VectorStoreIndex: Index based on Chroma vector store.
    """
    if source == "url":
        corpus_name = source_path.split("/")[-1] if "://" in source_path else source_path.replace(":", "_")
        data_dir = os.path.join("data", corpus_name)
        chroma_path = os.path.join("storage", "chroma", corpus_name)
    else:
        data_dir = os.path.join("data", "chroma")
        chroma_path = os.path.join("storage", "chroma", "local")

    os.makedirs(chroma_path, exist_ok=True)

    client = PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="chroma_collection")
    chroma_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=chroma_store)

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(
            vector_store=chroma_store,
            embed_model=embedding_model,
            storage_context=storage_context
        )
    else:
        documents = SimpleDirectoryReader(data_dir).load_data()
        return VectorStoreIndex.from_documents(
            documents,
            embed_model=embedding_model,
            storage_context=storage_context
        )


# === File: vector_db/simple_index.py ===
import os
from typing import Tuple
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utility.logger import logger
import requests
from datasets import load_dataset


def parse_source_path(source_path: str) -> Tuple[str, str]:
    """
    Parses the source path to determine its type and extract relevant information.

    Args:
        source_path (str): The source path to parse.

    Returns:
        Tuple[str, str]: A tuple containing the source type ('url' or 'hf') and the parsed corpus or dataset configuration.

    Raises:
        ValueError: If the source path format is unsupported.
    """
    logger.info(f"Parsing source path: {source_path}")
    if source_path.startswith("http://") or source_path.startswith("https://"):
        corpus_name = source_path.split("/")[-1]
        logger.info(f"Source type identified as URL with corpus name: {corpus_name}")
        return "url", corpus_name
    elif ":" in source_path:
        dataset_config = source_path.replace(":", "_")
        logger.info(f"Source type identified as Hugging Face dataset with config: {dataset_config}")
        return "hf", dataset_config
    else:
        logger.error(f"Unsupported source path format: {source_path}")
        raise ValueError(f"Unsupported source path format: {source_path}")


def validate_url(url: str) -> bool:
    """
    Validates the URL to ensure it is safe and matches expected patterns.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.

    Raises:
        ValueError: If the URL is invalid or unsafe.
    """
    logger.info(f"Validating URL: {url}")
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        logger.error("URL must use HTTPS.")
        raise ValueError("URL must use HTTPS.")
    allowed_domains = ["example.com", "trusted-source.com"]
    if parsed_url.netloc not in allowed_domains:
        logger.error(f"Domain '{parsed_url.netloc}' is not allowed.")
        raise ValueError(f"Domain '{parsed_url.netloc}' is not allowed.")
    return True


def download_and_save_from_url(url: str, target_dir: str) -> None:
    """
    Downloads text data from a URL and saves it locally using streaming.

    Args:
        url (str): URL to download the data from.
        target_dir (str): Directory to save the downloaded data.

    Returns:
        None
    """
    validate_url(url)
    logger.info(f"Downloading data from URL: {url}")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, "corpus.txt")

    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            logger.error(f"Failed to download from {url}, status code: {response.status_code}")
            raise Exception(f"Failed to download from {url}, status code: {response.status_code}")
        with open(file_path, "w", encoding="utf-8") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk.decode("utf-8"))
    logger.info(f"Downloaded and saved corpus to {file_path}")


def download_and_save_from_hf(dataset_name: str, config: str, target_dir: str, max_docs: int = 1000) -> None:
    """
    Downloads a dataset from Hugging Face and saves it locally as text files in batches.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        config (str): Configuration for the dataset.
        target_dir (str): Directory to save the downloaded documents.
        max_docs (int): Maximum number of documents to download.

    Returns:
        None
    """
    logger.info(f"Downloading dataset '{dataset_name}' with config '{config}' from Hugging Face...")
    dataset = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
    os.makedirs(target_dir, exist_ok=True)

    batch_size = 100
    batch = []
    for i, item in enumerate(dataset):
        if i >= max_docs:
            break
        text = item["text"].strip()
        if text:
            batch.append((i, text))
        if len(batch) == batch_size or i == max_docs - 1:
            for doc_id, doc_text in batch:
                with open(os.path.join(target_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(doc_text)
            batch.clear()
    logger.info(f"Saved {min(max_docs, len(dataset))} documents to {target_dir}")


def build_simple_vector_db(source: str, source_path: str, embedding_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Build a simple vector database vector_db using GPTVectorStoreIndex with persistent storage.

    Args:
        source (str): 'local' or 'url'
        source_path (str): Path or identifier for the source.
        embedding_model: Embedding model to use.

    Returns:
        VectorStoreIndex: The loaded or newly created vector store vector_db.
    """
    if source == "url":
        if source_path is None:
            logger.error("source_path must be provided for 'url' source.")
            raise ValueError("source_path must be provided for 'url' source.")

        source_type, corpus_name = parse_source_path(source_path)
        corpus_dir = os.path.join("data", corpus_name)
        storage_dir = os.path.join("storage", "simple", corpus_name)

        if not os.path.exists(corpus_dir):
            logger.info(f"Downloading corpus into {corpus_dir}...")
            if source_type == "hf":
                dataset, config = source_path.split(":", 1)
                download_and_save_from_hf(dataset, config, corpus_dir)
            else:
                download_and_save_from_url(source_path, corpus_dir)

    else:
        corpus_dir = os.path.join("data", "simple")
        storage_dir = os.path.join("storage", "simple")

    os.makedirs(storage_dir, exist_ok=True)

    if os.path.exists(storage_dir) and os.path.exists(os.path.join(storage_dir, "docstore.json")):
        logger.info(f"Loading existing simple vector database from {storage_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context, embed_model=embedding_model)

    logger.info(f"Indexing documents from {corpus_dir}...")
    documents = SimpleDirectoryReader(corpus_dir).load_data()
    vector_db = VectorStoreIndex.from_documents(
        documents,
        embed_model=embedding_model
    )
    vector_db.storage_context.persist(persist_dir=storage_dir)
    logger.info(f"Indexed {len(documents)} documents and saved to {storage_dir}.")
    return vector_db


# === File: modules/query.py ===
# modules/query.py - Refactored for better organization
import numpy as np
from typing import Union, Optional, Dict, Callable, List, Tuple, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, PreTrainedModel, PreTrainedTokenizer

from configurations.config import MAX_RETRIES, QUALITY_THRESHOLD, MAX_NEW_TOKENS
from utility.embedding_utils import get_query_vector
from utility.logger import logger
from utility.similarity_calculator import calculate_similarities, SimilarityMethod

# Import performance monitoring and caching
try:
    from utility.performance import monitor_performance, track_performance
    from utility.cache import cache_query_result

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


    def cache_query_result(model_name):
        def decorator(func):
            return func

        return decorator


# =====================================================
# EMBEDDING AND TEXT PROCESSING UTILITIES
# =====================================================

def extract_node_text_for_embedding(node) -> str:
    """
    Extract text from a LlamaIndex node exactly as LlamaIndex does for embedding.

    Args:
        node: LlamaIndex node object

    Returns:
        str: Text formatted for embedding, matching LlamaIndex's internal process
    """
    try:
        return node.get_content(metadata_mode=MetadataMode.EMBED)
    except Exception as e:
        logger.warning(f"Failed to extract embedding text: {e}. Using fallback.")
        return getattr(node, 'text', str(node))


def generate_embedding_with_normalization(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Generate embedding for text using LlamaIndex's method with optional normalization.

    Args:
        text: Text to embed
        embed_model: HuggingFace embedding model

    Returns:
        np.ndarray: Normalized embedding vector
    """
    with monitor_performance("embedding_generation"):
        logger.debug(f"Generating embedding for text: {text[:50]}...")

        embedding = embed_model.get_text_embedding(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        # Apply normalization if needed
        norm = np.linalg.norm(embedding_array)
        if norm > 1.1 or norm < 0.9:  # Not normalized
            embedding_array = embedding_array / norm
            logger.debug("Applied normalization to embedding")

        return embedding_array


def process_retrieved_nodes(nodes_with_scores) -> Tuple[List, List[float]]:
    """
    Extract nodes and scores from LlamaIndex retrieval results.

    Args:
        nodes_with_scores: LlamaIndex retrieval results

    Returns:
        Tuple of (nodes_list, scores_list)
    """
    nodes = [item.node for item in nodes_with_scores]
    scores = [item.score for item in nodes_with_scores]
    return nodes, scores


def batch_generate_embeddings(texts: List[str], embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of texts to embed
        embed_model: HuggingFace embedding model

    Returns:
        np.ndarray: Array of embeddings (n_texts, embedding_dim)
    """
    with monitor_performance("batch_embedding_generation"):
        embeddings = [
            generate_embedding_with_normalization(text, embed_model)
            for text in texts
        ]
        return np.array(embeddings)


# =====================================================
# SIMILARITY AND FILTERING UTILITIES
# =====================================================

def calculate_similarity_scores(
        query_vector: np.ndarray,
        document_embeddings: np.ndarray,
        method: Union[SimilarityMethod, str, Callable],
        reference_scores: Optional[List[float]] = None
) -> np.ndarray:
    """
    Calculate similarity scores and optionally compare with reference scores.

    Args:
        query_vector: Query embedding vector
        document_embeddings: Document embedding matrix
        method: Similarity calculation method
        reference_scores: Optional reference scores for comparison

    Returns:
        np.ndarray: Calculated similarity scores
    """
    with monitor_performance("similarity_calculation"):
        similarities = calculate_similarities(query_vector, document_embeddings, method)

        # Log comparison if using cosine similarity and reference scores available
        if method == SimilarityMethod.COSINE and reference_scores is not None:
            _log_similarity_comparison(similarities, reference_scores)

        return similarities


def _log_similarity_comparison(manual_scores: np.ndarray, reference_scores: List[float]):
    """Log comparison between manual and reference similarity scores."""
    logger.info("Similarity score comparison:")
    for i, (manual, reference) in enumerate(zip(manual_scores, reference_scores)):
        diff = abs(manual - reference)
        logger.info(f"Doc {i}: Manual={manual:.6f}, Reference={reference:.6f}, Diff={diff:.6f}")


def filter_and_rank_results(
        similarity_scores: np.ndarray,
        node_contents: List[str],
        similarity_threshold: float,
        max_results: int
) -> List[str]:
    """
    Filter results by similarity threshold and return top-k ranked by score.

    Args:
        similarity_scores: Array of similarity scores
        node_contents: List of node content strings
        similarity_threshold: Minimum similarity score to include
        max_results: Maximum number of results to return

    Returns:
        List[str]: Filtered and ranked content strings
    """
    with monitor_performance("result_filtering_and_ranking"):
        # Vectorized filtering
        valid_mask = similarity_scores >= similarity_threshold

        if not np.any(valid_mask):
            logger.warning(f"No results above similarity threshold {similarity_threshold}")
            return []

        # Extract valid results
        valid_scores = similarity_scores[valid_mask]
        valid_contents = [content for i, content in enumerate(node_contents) if valid_mask[i]]

        # Sort by score (descending) and take top-k
        sorted_indices = np.argsort(valid_scores)[::-1][:max_results]
        ranked_contents = [valid_contents[i] for i in sorted_indices]

        logger.info(f"Filtered to {len(ranked_contents)} results from {len(node_contents)} candidates")
        return ranked_contents


# =====================================================
# CORE RETRIEVAL FUNCTIONS
# =====================================================

@track_performance("context_retrieval")
def retrieve_context_with_similarity(
        query: Union[str, np.ndarray],
        vector_db: VectorStoreIndex,
        embed_model: HuggingFaceEmbedding,
        max_results: int = 5,
        similarity_threshold: float = 0.5,
        similarity_method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE,
) -> str:
    """
    Retrieve relevant context using configurable similarity methods.

    Args:
        query: Query string or vector
        vector_db: LlamaIndex vector store
        embed_model: HuggingFace embedding model
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score threshold
        similarity_method: Method for calculating similarity

    Returns:
        str: Concatenated context from relevant documents
    """
    logger.info(f"Retrieving context using {similarity_method} similarity")

    # Input validation
    if not isinstance(query, (str, np.ndarray)):
        raise TypeError("Query must be a string or numpy array")

    # Step 1: Initial retrieval using LlamaIndex
    nodes_with_scores = _perform_initial_retrieval(query, vector_db, max_results)
    if not nodes_with_scores:
        return ''

    # Step 2: Process query and document embeddings
    query_vector = _prepare_query_vector(query, embed_model)
    nodes, reference_scores = process_retrieved_nodes(nodes_with_scores)

    # Step 3: Generate document embeddings
    document_embeddings = _generate_document_embeddings(nodes, embed_model)

    # Step 4: Calculate similarities using specified method
    similarity_scores = calculate_similarity_scores(
        query_vector, document_embeddings, similarity_method, reference_scores
    )

    # Step 5: Filter, rank, and format results
    node_contents = [node.get_content() for node in nodes]
    filtered_contents = filter_and_rank_results(
        similarity_scores, node_contents, similarity_threshold, max_results
    )

    logger.info(f"Retrieved {len(filtered_contents)} relevant documents")
    return "\n\n".join(filtered_contents)  # Double newline for better separation


def _perform_initial_retrieval(query, vector_db, max_results):
    """Perform initial retrieval using LlamaIndex."""
    with monitor_performance("llamaindex_retrieval"):
        retriever = vector_db.as_retriever(similarity_top_k=max_results)
        nodes_with_scores = retriever.retrieve(query)

        if nodes_with_scores:
            logger.info(f"LlamaIndex retrieved {len(nodes_with_scores)} initial candidates")
        else:
            logger.warning("No documents retrieved by LlamaIndex")

        return nodes_with_scores


def _prepare_query_vector(query, embed_model):
    """Prepare query vector from string or array input."""
    if isinstance(query, str):
        if embed_model is None:
            raise ValueError("Embedding model required for string queries")
        with monitor_performance("query_embedding"):
            return get_query_vector(query, embed_model)
    return query


def _generate_document_embeddings(nodes, embed_model):
    """Generate embeddings for document nodes."""
    with monitor_performance("document_embedding_extraction"):
        texts = [extract_node_text_for_embedding(node) for node in nodes]
        return batch_generate_embeddings(texts, embed_model)


# =====================================================
# TEXT GENERATION UTILITIES
# =====================================================

# REPLACE YOUR EXISTING prepare_generation_inputs FUNCTION WITH THIS:
def prepare_generation_inputs(prompt: str, tokenizer, device: torch.device, max_length: int = 900):
    """
    MPS-safe input preparation.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # MPS-safe processing
    safe_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if device.type == "mps":
                if value.dtype == torch.bfloat16:
                    safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                elif key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    safe_inputs[key] = value.to(dtype=torch.long, device=device)
                else:
                    safe_inputs[key] = value.to(device)
            else:
                safe_inputs[key] = value.to(device)
        else:
            safe_inputs[key] = value

    return safe_inputs


def force_float32_on_mps_model(model):
    """Force all model parameters to float32 for MPS compatibility."""
    if next(model.parameters()).device.type == "mps":
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                print(f"Converting {name} from bfloat16 to float32")
                param.data = param.data.to(torch.float32)

        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.bfloat16:
                print(f"Converting buffer {name} from bfloat16 to float32")
                buffer.data = buffer.data.to(torch.float32)
    return model


# Fixed safe_mps_generate function - Remove torch_dtype parameter

def safe_mps_generate(model, tokenizer, inputs, device, max_tokens=64):
    """MPS-safe text generation that handles bfloat16 issues."""

    print(f"🔧 Called safe_mps_generate with device: {device}")  # Debug

    # Step 1: Force model to float32 if on MPS
    if device.type == "mps":
        print("🔧 Converting model to float32 for MPS")  # Debug
        model = force_float32_on_mps_model(model)

    # Step 2: Ensure all inputs are correct dtype
    safe_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if device.type == "mps":
                # For MPS, ensure proper dtypes
                if value.dtype == torch.bfloat16:
                    print(f"🔧 Converting input {key} from bfloat16 to float32")  # Debug
                    safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                elif key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    # These should be long tensors
                    safe_inputs[key] = value.to(dtype=torch.long, device=device)
                else:
                    # Default to float32 for MPS
                    if torch.is_floating_point(value):
                        safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                    else:
                        safe_inputs[key] = value.to(device)
            else:
                # For non-MPS devices, just move to device
                safe_inputs[key] = value.to(device)
        else:
            safe_inputs[key] = value

    # Debug: Print tensor dtypes
    print("🔍 Input tensor dtypes:")
    for key, value in safe_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype} on {value.device}")

    # Step 3: Set model to eval mode and disable gradients
    model.eval()

    try:
        with torch.no_grad():
            print(f"🚀 Starting generation on {device}")  # Debug

            # Use conservative generation settings for MPS
            if device.type == "mps":
                outputs = model.generate(
                    **safe_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding is more stable
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    # REMOVED: torch_dtype=torch.float32,  # This is not a valid parameter for generate()
                )
            else:
                # Standard generation for other devices
                outputs = model.generate(
                    **safe_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                )

        print("✅ Generation completed successfully")  # Debug
        return outputs

    except RuntimeError as e:
        print(f"❌ Generation failed: {e}")  # Debug

        if "bfloat16" in str(e).lower() or "mps" in str(e).lower():
            print(f"🔄 MPS generation failed, falling back to CPU: {e}")

            # Move to CPU and try again
            model_cpu = model.cpu()
            inputs_cpu = {k: v.cpu() for k, v in safe_inputs.items()}

            with torch.no_grad():
                outputs = model_cpu.generate(
                    **inputs_cpu,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Move model back to original device
            try:
                model.to(device)
                print(f"✅ Model moved back to {device}")
            except Exception as move_error:
                print(f"⚠️  Warning: Could not move model back to {device}: {move_error}")

            return outputs
        else:
            raise e


def generate_text_by_device(model, inputs, device, tokenizer, max_tokens: int = 64):
    """
    MPS-safe replacement for generate_text_by_device.
    """
    return safe_mps_generate(model, tokenizer, inputs, device, max_tokens)


def handle_gpu_memory_error(
        error: RuntimeError,
        model: AutoModelForCausalLM,
        inputs: dict,
        device: torch.device,
        tokenizer
) -> Tuple[str, str]:
    """
    Enhanced GPU memory error handling with MPS support.
    """
    logger.warning(f"GPU memory issue: {error}")

    # Clear GPU cache based on device type
    if device.type == "mps":
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

    # CPU fallback
    logger.info("Falling back to CPU generation")
    model_cpu = model.cpu()
    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_cpu.generate(
            **inputs_cpu,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Move model back to original device
    try:
        model.to(device)
        logger.info(f"Model moved back to {device}")
    except Exception as e:
        logger.warning(f"Failed to move model back to {device}: {e}")

    return answer, "cpu_fallback"


# =====================================================
# MAIN QUERY PROCESSING
# =====================================================

@track_performance("answer_quality_evaluation")
def evaluate_answer_quality(
        answer: str,
        question: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> float:
    """
    Evaluate the quality of a generated answer.

    Currently returns a placeholder score.
    TODO: Implement actual quality evaluation logic.
    """
    logger.info("Evaluating answer quality")
    # Placeholder - implement actual evaluation logic
    score = 1.0
    logger.info(f"Quality score: {score}")
    return score


@cache_query_result("distilgpt2")
@track_performance("complete_query_processing")
def process_query_with_context(
        prompt: str,
        model: Union[PreTrainedModel, Any],
        tokenizer: Union[PreTrainedTokenizer, Any],
        device: torch.device,
        vector_db: Optional[VectorStoreIndex] = None,
        embedding_model: Optional[HuggingFaceEmbedding] = None,
        max_retries: int = MAX_RETRIES,
        quality_threshold: float = QUALITY_THRESHOLD,
        similarity_method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> Dict[str, Union[str, float, int, None]]:
    """
    Process a query with optional context retrieval and iterative improvement.

    This is the main entry point for query processing with RAG capabilities.
    """
    logger.info(f"Processing query with {similarity_method} similarity")

    # Ensure model is on correct device
    _ensure_model_on_device(model, device)

    try:
        for attempt in range(max_retries + 1):
            try:
                # Generate answer for current attempt
                result = _generate_single_answer(
                    prompt, model, tokenizer, device, vector_db, embedding_model, similarity_method)

                # Check if answer meets quality threshold
                if result["score"] >= quality_threshold or attempt == max_retries:
                    result.update({
                        "attempts": attempt + 1,
                        "similarity_method": str(similarity_method),
                        "error": None,
                        "device_used": str(device),
                        "question": prompt,
                    })
                    logger.info("Query processing completed successfully")
                    return result

                # Improve prompt for next iteration
                prompt = _improve_prompt(prompt, result["answer"], model, tokenizer, device)

            except RuntimeError as err:
                if _is_gpu_memory_error(err):
                    answer, device_used = handle_gpu_memory_error(err, model, {}, device, tokenizer)
                    return _create_result_dict(prompt, answer, 1.0, attempt + 1,
                                               f"GPU fallback: {err}", device_used, similarity_method)
                else:
                    raise err

        # If we reach here, max retries exceeded
        return _create_result_dict(prompt, "No satisfactory answer generated", 0.0,
                                   max_retries + 1, None, str(device), similarity_method)

    except Exception as err:
        logger.error(f"Error during query processing: {err}")
        return _create_result_dict(prompt, f"Error: {err}", 0.0, 0, str(err), str(device), similarity_method)


def _ensure_model_on_device(model: AutoModelForCausalLM, device: torch.device):
    """Ensure model is on the correct device."""
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.info(f"Moving model from {model_device} to {device}")
        model.to(device)


@track_performance("single_answer_generating")
def _generate_single_answer(prompt, model, tokenizer, device, vector_db, embedding_model, similarity_method):
    """
    Updated single answer generation with MPS safety.
    """
    try:
        # Prepare prompt with context if available
        augmented_prompt = _prepare_prompt_with_context(
            prompt, vector_db, embedding_model, similarity_method
        )

        # Use MPS-safe input preparation
        inputs = prepare_generation_inputs(augmented_prompt, tokenizer, device)

        # Use MPS-safe generation
        outputs = generate_text_by_device(model, inputs, device, tokenizer)

        with monitor_performance("answer_processing"):
            outputs = outputs.cpu()
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = raw_output.split("Answer:")[-1].strip()
            score = evaluate_answer_quality(answer, prompt, model, tokenizer, device)

        return {
            "answer": answer,
            "score": score,
            "error": None,
            "device_used": str(device),
            "question": prompt,
            "similarity_method": str(similarity_method)
        }

    except RuntimeError as e:
        # Handle GPU memory errors specifically
        if _is_gpu_memory_error(e):
            logger.warning("GPU memory error detected, attempting fallback")
            answer, device_used = handle_gpu_memory_error(e, model, inputs, device, tokenizer)
            return {
                "answer": answer,
                "score": 1.0,  # Assume fallback worked
                "error": f"GPU fallback: {str(e)}",
                "device_used": device_used,
                "question": prompt,
                "similarity_method": str(similarity_method)
            }
        else:
            raise e  # Re-raise non-memory errors
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": "",
            "score": 0.0,
            "error": str(e),
            "device_used": str(device),
            "question": prompt,
            "similarity_method": str(similarity_method)
        }


@track_performance("preparing_query_context")
def _prepare_prompt_with_context(prompt, vector_db, embedding_model, similarity_method):
    """Prepare prompt with retrieved context if available."""
    original_question = prompt  # לשמור על שאלה מקורית

    if vector_db is not None:
        with monitor_performance("context_retrieval_and_prompt_construction"):
            context = retrieve_context_with_similarity(
                original_question, vector_db, embedding_model, similarity_method=similarity_method
            )
            if context and context.strip():
                return f"Context:\n{context}\n\nQuestion: {original_question}\nAnswer:"

    return f"Question: {original_question}\nAnswer:"


def _improve_prompt(original_prompt, previous_answer, model, tokenizer, device):
    """Improve prompt based on previous answer."""
    return rephrase_query(original_prompt, previous_answer, model, tokenizer, device)


def _is_gpu_memory_error(error):
    """Enhanced GPU memory error detection for both MPS and CUDA."""
    error_str = str(error).lower()
    gpu_memory_indicators = [
        "mps",
        "out of memory",
        "bfloat16",
        "memory",
        "allocation failed",
        "cuda out of memory"
    ]
    return any(indicator in error_str for indicator in gpu_memory_indicators)


def _create_result_dict(question, answer, score, attempts, error, device_used, similarity_method):
    """Create standardized result dictionary."""
    return {
        "question": question,
        "answer": answer,
        "score": score,
        "attempts": attempts,
        "error": error,
        "device_used": device_used,
        "similarity_method": str(similarity_method)
    }


@track_performance("query_rephrasing")
def rephrase_query(
        original_prompt: str,
        previous_answer: str,
        model: AutoModelForCausalLM,
        tokenizer,
        device: torch.device
) -> str:
    """
    MPS-safe query rephrasing.
    """
    logger.info("Rephrasing query for improvement")

    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"Previous answer: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )

    try:
        # Use MPS-safe input preparation
        inputs = prepare_generation_inputs(rephrase_prompt, tokenizer, device, max_length=800)

        with monitor_performance("rephrase_generation"):
            with torch.no_grad():
                # Use MPS-safe generation
                outputs = generate_text_by_device(model, inputs, device, tokenizer, max_tokens=MAX_NEW_TOKENS)

        outputs = outputs.cpu()
        rephrased_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info("Query rephrased successfully")
        return rephrased_query

    except Exception as e:
        logger.error(f"Query rephrasing failed: {e}")
        # Return original prompt if rephrasing fails
        return original_prompt


# Example usage and testing
def test_mps_query_processing():
    """Test the MPS-safe query processing pipeline."""
    print("🧪 Testing MPS-safe query processing...")

    try:
        from modules.model_loader import load_model
        from modules.indexer import load_vector_db
        from configurations.config import INDEX_SOURCE_URL

        # Load model
        tokenizer, model = load_model()
        device = next(model.parameters()).device
        print(f"✅ Model loaded on {device}")

        # Load vector DB (optional)
        try:
            vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
            print("✅ Vector DB loaded")
        except Exception as e:
            print(f"⚠️  Vector DB failed to load: {e}")
            vector_db, embedding_model = None, None

        # Test simple query
        test_query = "What is artificial intelligence?"
        print(f"\n🔍 Testing query: '{test_query}'")

        result = process_query_with_context(
            test_query, model, tokenizer, device,
            vector_db, embedding_model, max_retries=1
        )

        if result["error"]:
            print(f"❌ Query failed: {result['error']}")
            return False
        else:
            print(f"✅ Query successful!")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Score: {result['score']}")
            print(f"   Device: {result['device_used']}")
            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = test_mps_query_processing()
    print(f"\n🎯 MPS Query Processing Test: {'PASSED' if success else 'FAILED'}")


# === File: modules/indexer.py ===
# indexer.py
import os
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from configurations.config import HF_MODEL_NAME
from vector_db.chroma_index import build_chroma_vector_db
from vector_db.simple_index import build_simple_vector_db
from utility.device_utils import get_optimal_device
from utility.logger import logger
from typing import Optional
import torch


def load_vector_db(source: str = "local", source_path: Optional[str] = None, storing_method: str = "chroma") -> (
        VectorStoreIndex, HuggingFaceEmbedding):
    """
    Loads or creates a vector database for document retrieval with optimized embedding model caching.

    Args:
        source (str): Source type ('local' or 'url').
        source_path (Optional[str]): Path to the data source.
        storing_method (str): Storage method to use ('chroma' or 'llama_index').

    Returns:
        Tuple[VectorStoreIndex, HuggingFaceEmbedding]: Loaded or newly created vector database and embedding model.
    """
    logger.info(
        f"Loading vector database from source: {source}, source_path: {source_path}, storing_method: {storing_method}")

    device = get_optimal_device()  # Now you can directly call it

    # Initialize HuggingFaceEmbedding with explicit device and dtype for MPS
    if device.type == "mps":
        logger.info(f"Initializing HuggingFaceEmbedding for MPS with torch_dtype=torch.float32")
        embedding_model = HuggingFaceEmbedding(
            model_name=HF_MODEL_NAME,
            device=str(device),  # Pass the device as a string "mps"
            model_kwargs={"torch_dtype": torch.float32}
        )
    elif device.type == "cuda":
        # Ensure USE_MIXED_PRECISION is also imported or handled
        from configurations.config import USE_MIXED_PRECISION
        torch_dtype = torch.float16 if USE_MIXED_PRECISION else torch.float32
        logger.info(f"Initializing HuggingFaceEmbedding for CUDA with torch_dtype={torch_dtype}")
        embedding_model = HuggingFaceEmbedding(
            model_name=HF_MODEL_NAME,
            device=str(device),
            model_kwargs={"torch_dtype": torch_dtype}
        )
    else:  # CPU
        logger.info(f"Initializing HuggingFaceEmbedding for CPU")
        embedding_model = HuggingFaceEmbedding(
            model_name=HF_MODEL_NAME,
            device=str(device),
            model_kwargs={"torch_dtype": torch.float32}  # Always use float32 on CPU
        )

    if not hasattr(load_vector_db, "_embed_model"):
        load_vector_db._embed_model = embedding_model

    if storing_method == "chroma":
        vector_db = build_chroma_vector_db(source=source, source_path=source_path, embedding_model=embedding_model)
        return vector_db, embedding_model

    elif storing_method == "llama_index":
        vector_db = build_simple_vector_db(source=source, source_path=source_path, embedding_model=embedding_model)
        return vector_db, embedding_model

    else:
        logger.error(f"Unsupported storing method: {storing_method}")
        raise ValueError(f"Unsupported storing method: {storing_method}")


# === File: modules/model_loader.py ===
# modules/model_loader.py - Complete version with performance monitoring
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from configurations.config import MODEL_PATH, FORCE_CPU, OPTIMIZE_FOR_MPS, USE_MIXED_PRECISION
from typing import Tuple

from utility.device_utils import get_optimal_device
from utility.logger import logger

try:
    from utility.performance import monitor_performance, track_performance

    PERFORMANCE_AVAILABLE = True
except ImportError:
    logger.warning("Performance monitoring not available. Install with: pip install psutil")
    PERFORMANCE_AVAILABLE = False


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


def force_float32_recursive(model):
    """
    Recursively convert all model parameters to float32.
    This ensures no bfloat16 tensors remain in the model.
    """
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            logger.info(f"Converting {name} from bfloat16 to float32")
            param.data = param.data.to(torch.float32)

    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            logger.info(f"Converting buffer {name} from bfloat16 to float32")
            buffer.data = buffer.data.to(torch.float32)

    return model


def prepare_mps_inputs(inputs, device):
    """
    Prepare inputs for MPS with proper dtype handling.
    """
    prepared_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            # Force all tensors to be float32 or long for MPS compatibility
            if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
                if key in ['attention_mask', 'token_type_ids']:
                    # These should be long tensors
                    prepared_inputs[key] = value.to(dtype=torch.long, device=device)
                else:
                    # Convert to float32
                    prepared_inputs[key] = value.to(dtype=torch.float32, device=device)
            elif value.dtype in [torch.int32, torch.int64]:
                # Keep integer types as long
                prepared_inputs[key] = value.to(dtype=torch.long, device=device)
            else:
                # Default case
                prepared_inputs[key] = value.to(device=device)
        else:
            prepared_inputs[key] = value

    return prepared_inputs


@track_performance("complete_model_loading")
def load_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    logger.info(f"Loading Model {MODEL_PATH} with MPS compatibility...")
    device = get_optimal_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added padding token to tokenizer.")

    logger.info("Tokenizer loaded successfully.")

    try:
        if device.type == "mps":
            # Special handling for MPS
            logger.info("Loading model with MPS-specific optimizations...")

            # Load on CPU first to avoid MPS dtype issues during loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,  # Explicitly use float32
                low_cpu_mem_usage=True,
                device_map=None  # Don't auto-assign device
            )

            # Force all parameters to float32 recursively
            model = force_float32_recursive(model)

            # Now move to MPS
            model = model.to(device)

            # Verify no bfloat16 tensors remain
            bfloat16_params = [name for name, param in model.named_parameters()
                               if param.dtype == torch.bfloat16]
            if bfloat16_params:
                logger.warning(f"Found remaining bfloat16 parameters: {bfloat16_params}")

            logger.info("Model successfully loaded and converted for MPS")

        elif device.type == "cuda":
            torch_dtype = torch.float16 if USE_MIXED_PRECISION else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            logger.info(f"Loaded model for CUDA with {torch_dtype}")

        else:
            # CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            logger.info("Loaded model for CPU")

        # Skip torch.compile for MPS (known compatibility issues)
        if device.type != "mps" and torch.__version__.startswith("2"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model optimized with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        logger.info("Model loading completed successfully")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


def generate_text_mps_safe(model, inputs, device, tokenizer, max_tokens: int = 64):
    """
    MPS-safe text generation function.
    """
    try:
        # Prepare inputs for MPS
        if device.type == "mps":
            inputs = prepare_mps_inputs(inputs, device)

        # Ensure model is in eval mode
        model.eval()

        with torch.no_grad():
            # Use conservative generation parameters for MPS
            if device.type == "mps":
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding is more stable on MPS
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                # Use normal generation for other devices
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                )

        return generated_ids

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Try CPU fallback
        logger.info("Attempting CPU fallback...")
        model_cpu = model.cpu()
        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_cpu.generate(
                **inputs_cpu,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Move model back to original device
        model.to(device)
        return outputs


def test_mps_compatibility():
    """
    Test MPS compatibility with a simple generation.
    """
    logger.info("Testing MPS compatibility...")

    try:
        tokenizer, model = load_model()
        device = next(model.parameters()).device

        # Test with a simple prompt
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Generate response
        outputs = generate_text_mps_safe(model, inputs, device, tokenizer, max_tokens=10)

        # Decode result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"MPS test successful. Generated: {result}")

        return True

    except Exception as e:
        logger.error(f"MPS test failed: {e}")
        return False


def prepare_generation_inputs_mps_safe(prompt: str, tokenizer, device: torch.device, max_length: int = 900):
    """
    MPS-safe input preparation with proper dtype handling.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # Handle MPS-specific dtype requirements
    if device.type == "mps":
        return prepare_mps_inputs(inputs, device)
    else:
        # Standard device handling for CUDA/CPU
        normalized_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                normalized_inputs[k] = v.to(device, non_blocking=True)
            else:
                normalized_inputs[k] = v
        return normalized_inputs


def process_query_with_context_mps_safe(
        prompt: str,
        model,
        tokenizer,
        device: torch.device,
        vector_db=None,
        embedding_model=None,
        max_retries: int = 3,
        quality_threshold: float = 0.5
):
    """
    MPS-safe query processing with proper error handling.
    """
    logger.info(f"Processing query on {device}")

    # Prepare prompt with context if available
    if vector_db is not None and embedding_model is not None:
        # Your existing context retrieval logic here
        context = "Your retrieved context..."  # Replace with actual retrieval
        augmented_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    else:
        augmented_prompt = f"Question: {prompt}\nAnswer:"

    try:
        # Prepare inputs with MPS safety
        inputs = prepare_generation_inputs_mps_safe(augmented_prompt, tokenizer, device)

        # Generate with MPS-safe function
        outputs = generate_text_mps_safe(model, inputs, device, tokenizer)

        # Process output
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = raw_output.split("Answer:")[-1].strip()

        return {
            "question": prompt,
            "answer": answer,
            "error": None,
            "device_used": str(device),
            "score": 1.0  # Placeholder
        }

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "question": prompt,
            "answer": f"Error: {e}",
            "error": str(e),
            "device_used": str(device),
            "score": 0.0
        }


@track_performance("model_capabilities_analysis")
def get_model_capabilities(model) -> dict:
    with monitor_performance("capability_analysis"):
        try:
            first_param = next(iter(model.parameters()))
            device = str(first_param.device)
            dtype = str(first_param.dtype)
        except StopIteration:
            device = "unknown"
            dtype = "unknown"

        parameter_count = sum(p.numel() for p in model.parameters())

        return {
            'can_generate': hasattr(model, 'generate'),
            'model_type': type(model).__name__,
            'is_causal_lm': 'CausalLM' in type(model).__name__,
            'device': device,
            'dtype': dtype,
            'parameter_count': parameter_count,
        }


@track_performance("gpu_memory_monitoring")
def monitor_gpu_memory():
    if torch.backends.mps.is_available():
        logger.info("MPS: Memory monitoring limited on Apple Silicon")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(f"CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        logger.info("CPU: No GPU memory to monitor")


@track_performance("complete_model_test")
def test_model_loading():
    logger.info("Testing GPU-optimized model loading...")
    try:
        with monitor_performance("test_model_loading"):
            tokenizer, model = load_model()

        with monitor_performance("test_capabilities_analysis"):
            capabilities = get_model_capabilities(model)

        logger.info("Model loading test results:")
        for key, value in capabilities.items():
            status = "PASS" if value else "WARN" if key == 'is_causal_lm' else "INFO"
            logger.info(f"  {status} {key}: {value}")

        monitor_gpu_memory()

        if capabilities['can_generate']:
            logger.info("Model supports text generation")
            with monitor_performance("test_text_generation"):
                device = next(model.parameters()).device
                test_input = tokenizer("Hello", return_tensors="pt")

                # Use MPS-safe input preparation
                test_input = prepare_mps_inputs(test_input, device) if device.type == "mps" else {k: v.to(device) for
                                                                                                  k, v in
                                                                                                  test_input.items()}

                with torch.no_grad():
                    # Use MPS-safe generation
                    outputs = generate_text_mps_safe(
                        model, test_input, device, tokenizer, max_tokens=5
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Test generation on {device}: '{result}'")
            monitor_gpu_memory()
        else:
            logger.warning("Model does NOT support text generation")

        return capabilities['can_generate']

    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


def benchmark_model_performance(num_iterations: int = 5):
    logger.info(f"Starting model performance benchmark ({num_iterations} iterations)")
    results = {'loading_times': [], 'generation_times': [], 'total_times': []}

    for i in range(num_iterations):
        logger.info(f"Iteration {i + 1}/{num_iterations}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        import time
        start_cpu = time.time()

        with monitor_performance(f"benchmark_iteration_{i + 1}"):
            tokenizer, model = load_model()
            device = next(model.parameters()).device
            test_input = tokenizer("Hello world", return_tensors="pt")

            # Use MPS-safe input preparation
            if device.type == "mps":
                test_input = prepare_mps_inputs(test_input, device)
            else:
                test_input = {k: v.to(device) for k, v in test_input.items()}

            with torch.no_grad():
                # Use MPS-safe generation
                outputs = generate_text_mps_safe(
                    model, test_input, device, tokenizer, max_tokens=10
                )

        iteration_time = time.time() - start_cpu
        results['total_times'].append(iteration_time)
        logger.info(f"  Iteration {i + 1} completed in {iteration_time:.2f}s")

    benchmark_stats = {
        'iterations': num_iterations,
        'avg_time': sum(results['total_times']) / len(results['total_times']),
        'min_time': min(results['total_times']),
        'max_time': max(results['total_times']),
        'total_time': sum(results['total_times'])
    }

    logger.info("Benchmark Results:")
    for key, value in benchmark_stats.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value:.2f}s")

    return benchmark_stats


def get_device_info():
    info = {
        'cpu_count': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'optimal_device': str(get_optimal_device())
    }

    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            'cuda_compute_capability': torch.cuda.get_device_capability(0)
        })

    if torch.backends.mps.is_available():
        info.update({'mps_device': 'Apple Silicon GPU detected'})

    return info


def print_system_info():
    logger.info("System Information:")
    for key, value in get_device_info().items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Model path: {MODEL_PATH}")
    logger.info(f"  Force CPU: {FORCE_CPU}")
    logger.info(f"  Optimize for MPS: {OPTIMIZE_FOR_MPS}")
    logger.info(f"  Use mixed precision: {USE_MIXED_PRECISION}")


if __name__ == "__main__":
    # Run comprehensive test
    print("🚀 Running comprehensive MPS compatibility tests...")

    # Test basic compatibility
    basic_test = test_mps_compatibility()
    print(f"Basic MPS test: {'✅ PASSED' if basic_test else '❌ FAILED'}")

    # Test model loading with capabilities
    loading_test = test_model_loading()
    print(f"Model loading test: {'✅ PASSED' if loading_test else '❌ FAILED'}")

    # Test system info
    print("\n📊 System Information:")
    print_system_info()

    # Optional: Run benchmark (comment out if you want to skip)
    print("\n⏱️  Running performance benchmark...")
    try:
        benchmark_stats = benchmark_model_performance(num_iterations=3)
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")

    print("\n🎉 All tests completed!")


# === File: utility/similarity_calculator.py ===
# utility/similarity_calculator.py
import numpy as np
from typing import Callable, Union
from enum import Enum


class SimilarityMethod(Enum):
    """Enumeration of available similarity methods."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class SimilarityCalculator:
    """High-performance similarity calculator with extensible architecture."""

    def __init__(self):
        # Core methods with optimized implementations
        self._batch_methods = {
            SimilarityMethod.COSINE: self._cosine_batch,
            SimilarityMethod.DOT_PRODUCT: self._dot_batch,
            SimilarityMethod.EUCLIDEAN: self._euclidean_batch,
        }

        self._pairwise_methods = {
            SimilarityMethod.COSINE: self._cosine_pairwise,
            SimilarityMethod.DOT_PRODUCT: self._dot_pairwise,
            SimilarityMethod.EUCLIDEAN: self._euclidean_pairwise,
        }

        # For custom methods
        self._custom_methods = {}

    def calculate_batch_similarity(
            self,
            query_vector: np.ndarray,
            document_embeddings: np.ndarray,
            method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
    ) -> np.ndarray:
        """
        High-performance batch similarity calculation.

        Args:
            query_vector: 1D query vector
            document_embeddings: 2D array (n_docs, embedding_dim)
            method: Similarity method

        Returns:
            np.ndarray: Similarity scores for all documents
        """
        if callable(method):
            # Custom function - use pairwise calculation
            return np.array([method(query_vector, doc) for doc in document_embeddings])

        if isinstance(method, str):
            # Check custom methods first
            if method in self._custom_methods:
                return np.array([self._custom_methods[method](query_vector, doc) for doc in document_embeddings])
            method = SimilarityMethod(method)

        return self._batch_methods[method](query_vector, document_embeddings)

    def calculate_pairwise_similarity(
            self,
            vector1: np.ndarray,
            vector2: np.ndarray,
            method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
    ) -> float:
        """
        Calculate similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector
            method: Similarity method

        Returns:
            float: Similarity score
        """
        if callable(method):
            return method(vector1, vector2)

        if isinstance(method, str):
            if method in self._custom_methods:
                return self._custom_methods[method](vector1, vector2)
            method = SimilarityMethod(method)

        return self._pairwise_methods[method](vector1, vector2)

    def add_method(self, name: str, pairwise_func: Callable[[np.ndarray, np.ndarray], float]):
        """
        Add a custom similarity method.

        Args:
            name: Method name
            pairwise_func: Function that takes two vectors and returns similarity
        """
        self._custom_methods[name] = pairwise_func

    # Optimized batch methods
    def _cosine_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity."""
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0.0:
            return np.zeros(len(document_embeddings))

        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        zero_mask = doc_norms == 0.0
        doc_norms = np.where(zero_mask, 1.0, doc_norms)  # Avoid division by zero

        similarities = np.dot(document_embeddings, query_vector) / (doc_norms * query_norm)
        similarities[zero_mask] = 0.0

        return similarities

    def _dot_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized dot product."""
        return np.dot(document_embeddings, query_vector)

    def _euclidean_batch(self, query_vector: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance converted to similarity."""
        distances = np.linalg.norm(document_embeddings - query_vector, axis=1)
        return 1.0 / (1.0 + distances)

    # Pairwise methods
    def _cosine_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise cosine similarity."""
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return np.dot(vector1, vector2) / (norm1 * norm2)

    def _dot_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise dot product."""
        return np.dot(vector1, vector2)

    def _euclidean_pairwise(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Pairwise Euclidean distance to similarity."""
        distance = np.linalg.norm(vector1 - vector2)
        return 1.0 / (1.0 + distance)


# Global instance
_calculator = SimilarityCalculator()


def calculate_similarities(
        query_vector: np.ndarray,
        document_embeddings: np.ndarray,
        method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> np.ndarray:
    """
    Main function for batch similarity calculation.

    Args:
        query_vector: Query vector (1D)
        document_embeddings: Document embeddings (2D)
        method: Similarity method

    Returns:
        np.ndarray: Similarity scores
    """
    return _calculator.calculate_batch_similarity(query_vector, document_embeddings, method)


def calculate_similarity(
        vector1: np.ndarray,
        vector2: np.ndarray,
        method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> float:
    """
    Main function for pairwise similarity calculation.

    Args:
        vector1: First vector
        vector2: Second vector
        method: Similarity method

    Returns:
        float: Similarity score
    """
    return _calculator.calculate_pairwise_similarity(vector1, vector2, method)


def add_similarity_method(name: str, func: Callable[[np.ndarray, np.ndarray], float]):
    """
    Add a custom similarity method globally.

    Args:
        name: Method name
        func: Pairwise similarity function
    """
    _calculator.add_method(name, func)


# === File: utility/embedding_utils.py ===
# utility/embedding_utils.py
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import performance monitoring and caching
try:
    from utility.cache import cache_embedding
    from utility.performance import monitor_performance

    CACHE_AVAILABLE = True
    PERFORMANCE_AVAILABLE = True
except ImportError as e:
    CACHE_AVAILABLE = False
    PERFORMANCE_AVAILABLE = False


    def cache_embedding(model_name):
        def decorator(func):
            return func

        return decorator


    def monitor_performance(name):
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield

        return dummy_context()


@cache_embedding("sentence-transformers/all-MiniLM-L6-v2")
def get_query_vector(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a query's text into a vector embedding using the specified embedding model.
    Now includes caching and performance monitoring.

    Args:
        text (str): The input query text.
        embed_model (HuggingFaceEmbedding): The embedding model used for generating embeddings.

    Returns:
        np.ndarray: The vector embedding of the query text.
    """
    with monitor_performance("embedding_generation"):
        vector = embed_model.get_query_embedding(text)
    return np.array(vector, dtype=np.float32)


def get_text_embedding(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Get text embedding for documents (not queries).

    Args:
        text (str): The input text.
        embed_model (HuggingFaceEmbedding): The embedding model.

    Returns:
        np.ndarray: The vector embedding of the text.
    """
    with monitor_performance("text_embedding_generation"):
        vector = embed_model.get_text_embedding(text)
    return np.array(vector, dtype=np.float32)


# === File: utility/cache.py ===
# utility/cache.py
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np
from utility.logger import logger


class SmartCache:
    """Intelligent caching system for RAG operations"""

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories too
        except Exception as err:
            logger.warning(f"Failed to create cache directory {cache_dir}: {err}")
            # Fallback to a simple cache directory in current path
            self.cache_dir = Path("temp_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text())
            except IOError as IOE:
                logger.warning("Cache metadata corrupted, starting fresh")
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    def _hash_key(self, key: str) -> str:
        """Create hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired"""
        if cache_key not in self.metadata:
            return True
        created_time = self.metadata[cache_key].get('created', 0)
        return time.time() - created_time > self.ttl_seconds

    def get(self, key: str, default=None) -> Any:
        """Get value from cache"""
        cache_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists() or self._is_expired(cache_key):
            logger.debug(f"Cache miss: {key[:50]}...")
            return default

        try:
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Cache hit: {key[:50]}...")
            return value
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return default

    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            self.metadata[cache_key] = {
                'original_key': key[:100],  # Store first 100 chars for debugging
                'created': time.time(),
                'size_bytes': cache_file.stat().st_size
            }
            self._save_metadata()
            logger.debug(f"Cached: {key[:50]}...")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cache"""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata = {}
        self._save_metadata()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size = sum(item.get('size_bytes', 0) for item in self.metadata.values())
        return {
            'entries': len(self.metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': str(self.cache_dir)
        }


class EmbeddingCache(SmartCache):
    """Specialized cache for embeddings"""

    def __init__(self, cache_dir: str = "cache/embeddings", ttl_seconds: int = 86400):  # 24 hour TTL
        super().__init__(cache_dir, ttl_seconds)

    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = f"embedding:{model_name}:{text}"
        return self.get(key)

    def set_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding"""
        key = f"embedding:{model_name}:{text}"
        self.set(key, embedding)


class QueryCache(SmartCache):
    """Specialized cache for query results"""

    def __init__(self, cache_dir: str = "cache/queries", ttl_seconds: int = 1800):  # 30 minute TTL
        super().__init__(cache_dir, ttl_seconds)

    def get_query_result(self, query: str, model_name: str, context_hash: str = "") -> Optional[Dict]:
        """Get cached query result"""
        key = f"query:{model_name}:{context_hash}:{query}"
        return self.get(key)

    def set_query_result(self, query: str, model_name: str, result: Dict, context_hash: str = ""):
        """Cache query result"""
        key = f"query:{model_name}:{context_hash}:{query}"
        self.set(key, result)


# Global cache instances with error handling
try:
    embedding_cache = EmbeddingCache()
    query_cache = QueryCache()
    general_cache = SmartCache()
    CACHE_INITIALIZED = True
except Exception as e:
    logger.warning(f"Failed to initialize caches: {e}")


    # Create dummy cache objects that don't actually cache
    class DummyCache:
        def get(self, key, default=None):
            return default

        def set(self, key, value):
            pass

        def clear(self):
            pass

        def get_stats(self):
            return {'entries': 0, 'total_size_mb': 0.0, 'cache_dir': 'disabled'}

        def get_embedding(self, text, model_name):
            return None

        def set_embedding(self, text, model_name, embedding):
            pass

        def get_query_result(self, query, model_name, context_hash=""):
            return None

        def set_query_result(self, query, model_name, result, context_hash=""):
            pass


    embedding_cache = DummyCache()
    query_cache = DummyCache()
    general_cache = DummyCache()
    CACHE_INITIALIZED = False


def cache_embedding(model_name: str):
    """Decorator to cache embedding results"""

    def decorator(func):
        def wrapper(text: str, *args, **kwargs):
            # Check cache first
            cached = embedding_cache.get_embedding(text, model_name)
            if cached is not None:
                return cached

            # Generate embedding
            result = func(text, *args, **kwargs)

            # Cache result
            if isinstance(result, np.ndarray):
                embedding_cache.set_embedding(text, model_name, result)

            return result

        return wrapper

    return decorator


def cache_query_result(model_name: str):
    """Decorator to cache query results"""

    def decorator(func):
        def wrapper(query: str, *args, **kwargs):
            # Create context hash from args (simplified)
            context_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]

            # Check cache first
            cached = query_cache.get_query_result(query, model_name, context_hash)
            if cached is not None:
                return cached

            # Generate result
            result = func(query, *args, **kwargs)

            # Cache result
            if isinstance(result, dict) and 'answer' in result:
                query_cache.set_query_result(query, model_name, result, context_hash)

            return result

        return wrapper

    return decorator


def cache_stats():
    """Print cache statistics"""
    if not CACHE_INITIALIZED:
        print("\nCACHE STATISTICS")
        print("=" * 40)
        print("Cache system is disabled")
        return

    print("\nCACHE STATISTICS")
    print("=" * 40)

    embedding_stats = embedding_cache.get_stats()
    query_stats = query_cache.get_stats()
    general_stats = general_cache.get_stats()

    print(f"Embeddings: {embedding_stats['entries']} entries, {embedding_stats['total_size_mb']:.1f}MB")
    print(f"Queries: {query_stats['entries']} entries, {query_stats['total_size_mb']:.1f}MB")
    print(f"General: {general_stats['entries']} entries, {general_stats['total_size_mb']:.1f}MB")

    total_mb = embedding_stats['total_size_mb'] + query_stats['total_size_mb'] + general_stats['total_size_mb']
    print(f"Total Cache: {total_mb:.1f}MB")


def clear_all_caches():
    """Clear all caches"""
    if not CACHE_INITIALIZED:
        logger.info("Cache system is disabled, nothing to clear")
        return

    embedding_cache.clear()
    query_cache.clear()
    general_cache.clear()
    logger.info("All caches cleared")


# === File: utility/logger.py ===
import logging

# Configure logger
logger = logging.getLogger("ModularRAGOptimization")
logger.setLevel(logging.INFO)

# # Stream handler for console output
# stream_handler = logging.StreamHandler()
# stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# stream_handler.setFormatter(stream_formatter)

# File handler for logging to a file
file_handler = logging.FileHandler("logger.log")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
# logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# === File: utility/device_utils.py ===
# utility/device_utils.py
import torch
from configurations.config import FORCE_CPU, OPTIMIZE_FOR_MPS  # Assuming these are in config

# You might need to adjust how logger is imported if it's not globally available
# For now, let's assume it's imported or passed if needed
from utility.logger import logger  # Assuming logger is always available here


def get_optimal_device():
    if FORCE_CPU:
        logger.info("CPU forced via config")
        return torch.device("cpu")

    if torch.backends.mps.is_available() and OPTIMIZE_FOR_MPS:
        logger.info("MPS (Apple Silicon GPU) detected and enabled")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("CUDA GPU detected")
        return torch.device("cuda")
    else:
        logger.info("Using CPU (no GPU available)")
        return torch.device("cpu")


# === File: utility/performance.py ===
# utility/performance.py
import time
import psutil
import torch
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List
import json
from utility.logger import logger


class PerformanceMonitor:
    """Real-time performance tracking for RAG operations"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_session = {}

    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.current_session[operation_name] = {
            'start_time': time.time(),
            'start_memory': self.get_memory_usage(),
            'start_gpu_memory': self.get_gpu_memory()
        }

    def end_operation(self, operation_name: str) -> Dict:
        """End timing and record metrics"""
        if operation_name not in self.current_session:
            logger.warning(f"Operation {operation_name} was never started")
            return {}

        session = self.current_session[operation_name]
        duration = time.time() - session['start_time']
        memory_delta = self.get_memory_usage() - session['start_memory']
        gpu_memory_delta = self.get_gpu_memory() - session['start_gpu_memory']

        metrics = {
            'operation': operation_name,
            'duration': duration,
            'memory_delta_mb': memory_delta,
            'gpu_memory_delta_mb': gpu_memory_delta,
            'timestamp': time.time()
        }

        self.metrics[operation_name].append(metrics)
        del self.current_session[operation_name]

        logger.info(
            f"Performance {operation_name}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB, GPU: {gpu_memory_delta:+.1f}MB")
        return metrics

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.backends.mps.is_available():
            # MPS doesn't have detailed memory reporting
            return 0.0
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def get_summary(self) -> Dict:
        """Get performance summary"""
        summary = {}
        for operation, measurements in self.metrics.items():
            if measurements:
                durations = [m['duration'] for m in measurements]
                summary[operation] = {
                    'count': len(measurements),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        return summary

    def save_metrics(self, filepath: str = "performance_metrics.json"):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        logger.info(f"Performance metrics saved to {filepath}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(operation_name: str):
    """Context manager for monitoring performance"""
    performance_monitor.start_operation(operation_name)
    try:
        yield
    finally:
        performance_monitor.end_operation(operation_name)


def performance_report():
    """Print a performance report"""
    summary = performance_monitor.get_summary()

    print("\nPERFORMANCE REPORT")
    print("=" * 50)

    for operation, stats in summary.items():
        print(f"\nOperation: {operation}")
        print(f"   Count: {stats['count']}")
        print(f"   Average: {stats['avg_duration']:.2f}s")
        print(f"   Min: {stats['min_duration']:.2f}s")
        print(f"   Max: {stats['max_duration']:.2f}s")
        print(f"   Total: {stats['total_duration']:.2f}s")

    return summary


# Decorator for automatic performance monitoring
def track_performance(operation_name: str = None):
    """Decorator to automatically track function performance"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with monitor_performance(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


