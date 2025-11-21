"""
User interface utilities for selecting vector database configurations
"""
import os
import sys
import termios
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Callable

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from configurations.config import NQ_SAMPLE_SIZE, MAX_NEW_TOKENS, TEMPERATURE, INDEX_SOURCE_URL, RETRIEVER_TOP_K, \
    ACTIVE_QA_DATASET
from configurations.config import TRILATERATION_ITERATIVE, TRILATERATION_MAX_REFINES, TRILATERATION_CONVERGENCE_TOL
from experiments.noise_experiment import run_noise_experiment
from experiments.retrieval_experiment import run_retrieval_base_experiment
from matrics.results_logger import ResultsLogger
from modules.model_loader import load_model
from modules.query import process_query_with_context
from scripts.evaluator import enumerate_top_documents
from scripts.qa_data_set_loader import load_qa_queries
from utility.device_utils import get_optimal_device
from utility.logger import logger
from vector_db.indexer import get_embedding_model_info, load_vector_db
from vector_db.storing_methods import StoringMethod
from vector_db.vector_db_interface import VectorDBInterface

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_dataset_and_corpus():
    from configurations.config import QA_DATASETS
    print_section_header("🎓 DATASET SELECTION")

    # Add manual mode at the top
    print("0. ❌ No Dataset (manual queries only)")
    dataset_keys = list(QA_DATASETS.keys())
    for i, key in enumerate(dataset_keys, 1):
        entry = QA_DATASETS[key]
        desc = entry.get("description", "")
        corpus = entry.get("corpus_source", "❌ None defined")
        print(f"{i}. {key} - {desc}\n   ↳ Corpus: {corpus}")

    options = [(str(i + 1), k) for i, k in enumerate(dataset_keys)]
    from configurations.config import ACTIVE_QA_DATASET
    default_label = f"(default: {ACTIVE_QA_DATASET})" if ACTIVE_QA_DATASET else "(default: No Dataset)"
    selected = ask_selection(
        prompt_text=f"\nSelect dataset (0-{len(dataset_keys)}) {default_label}: ",
        options=options,
        default=None
    )
    if not selected or selected.strip() == "":
        if ACTIVE_QA_DATASET:
            print(f"\n✅ Using default dataset from configuration: {ACTIVE_QA_DATASET}")
            dataset_key = ACTIVE_QA_DATASET
            dataset_cfg = QA_DATASETS.get(dataset_key, {})
            corpus_source = dataset_cfg.get("corpus_source")
            if corpus_source:
                print(f"📚 Using predefined corpus: {corpus_source}")
            else:
                print("⚠️ No corpus defined for this dataset. You will need to configure a Vector DB manually.")
            return dataset_key, corpus_source
        else:
            print("\n⚠️ No default dataset configured. Falling back to manual mode.")
            return None, None

    if selected == "manual" or selected == "0":
        print("\n⚠️ Manual mode selected — no predefined dataset or corpus.")
        print("You can still use a custom vector database or run without one.")
        return None, None

    # Otherwise, selected is a dataset key
    # If selected is a number, map to dataset_keys index
    try:
        idx = int(selected) - 1
        dataset_key = dataset_keys[idx]
    except (ValueError, IndexError):
        dataset_key = selected
    dataset_cfg = QA_DATASETS[dataset_key]
    corpus_source = dataset_cfg.get("corpus_source")

    print(f"\n✅ Selected dataset: {dataset_key}")
    if corpus_source:
        print(f"📚 Using predefined corpus: {corpus_source}")
    else:
        print("⚠️ No corpus defined for this dataset. You will need to configure a Vector DB manually.")

    return dataset_key, corpus_source


############################################################
# Special Command Handling (command-line and interactive)   #
############################################################

def print_help():
    print("\n📖 Available Special Commands:")
    print("  🆘 --help           - Show this help message")
    print("  🧹 --clear-cache    - Clear all caches")
    print("  📈 --performance    - Show performance report")
    print("  💾 --cache-stats    - Show cache statistics")
    print("  🧑‍🔬 --analyze       - Run analysis mode")
    print("  🚪 exit/quit/q      - Exit or return to main menu (where applicable)")
    print()


def handle_command(command: str) -> bool:
    """
    Handle special commands interactively or from arguments.
    Returns True if a special command was handled.
    """
    cmd = command.strip().lower()
    if cmd in ("--help", "help"):
        print_help()
        return True
    if cmd == "--clear-cache":
        if CACHE_AVAILABLE:
            clear_all_caches()
            print("🧹 All caches cleared.")
        else:
            print("⚠️  Cache clearing not available.")
        return True
    if cmd == "--performance":
        if PERFORMANCE_AVAILABLE:
            performance_report()
        else:
            print("⚠️  Performance monitoring not available.")
        return True
    if cmd == "--cache-stats":
        if CACHE_AVAILABLE:
            cache_stats()
        else:
            print("⚠️  Cache statistics not available.")
        return True
    if cmd == "--analyze":
        print("🧑‍🔬 Entering analysis mode...")
        # Defer to analysis mode in main program, just signal handled.
        # Actual mode switch logic should be in the main loop.
        return True
    return False


def _wrap_special_commands(fn):
    """
    Decorator to wrap input prompts to handle special commands.
    If a special command is entered, handle it, then re-prompt.
    """

    def wrapper(*args, **kwargs):
        while True:
            result = fn(*args, **kwargs)
            # Only handle str results
            if isinstance(result, str) and handle_command(result):
                continue
            return result

    return wrapper


class Mode(str, Enum):
    INTERACTIVE = "1"
    EVALUATION = "2"
    DEVELOPMENT = "3"
    ANALYSIS = "4"
    INFO = "5"
    DOWNLOAD = "6"
    EXIT = "7"


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


def prompt_with_validation(
        prompt_text: str,
        validation_fn: Callable[[str], bool],
        default: Optional[str] = None,
        transform_fn: Optional[Callable[[str], str]] = None,
        error_msg: str = "❌ Invalid input."
) -> str:
    while True:
        user_input = input(prompt_text).strip()

        # Allow empty input to propagate upward only if no default is provided
        if not user_input:
            if default is None:
                # Let higher-level logic handle empty input
                return ""
            else:
                print(f"✅ Using default: {default}")
                return default

        # Remove surrounding quotes if present
        if user_input.startswith('"') and user_input.endswith('"'):
            user_input = user_input[1:-1].strip()
        elif user_input.startswith("'") and user_input.endswith("'"):
            user_input = user_input[1:-1].strip()

        try:
            normalized_input = user_input.lower()
            if validation_fn(normalized_input):
                result = transform_fn(normalized_input) if transform_fn else normalized_input
                print(f"✅ Selected: {result}")
                return result
        except Exception:
            pass

        print(error_msg)


@_wrap_special_commands
def ask_selection(prompt_text: str, options: list, default: Optional[str] = None) -> str:
    """
    Prompt the user to select one of the provided options by name or index (1-based).

    Args:
        prompt_text (str): The prompt to display.
        options (list): List of string options or (value, label) pairs.
        default (Optional[str]): Default value to use if user presses Enter.

    Returns:
        str: The selected option's value.
    """
    if all(isinstance(opt, tuple) for opt in options):
        value_label_pairs = options
    else:
        value_label_pairs = [(opt.lower(), opt) for opt in options]

    valid_values = [str(v) for v, _ in value_label_pairs]
    index_map = {str(i + 1): v for i, (v, _) in enumerate(value_label_pairs)}

    def validate(user_input: str) -> bool:
        return user_input.lower() in valid_values or user_input in index_map

    def transform(user_input: str) -> str:
        return index_map[user_input] if user_input in index_map else user_input.lower()

    return prompt_with_validation(
        prompt_text=prompt_text,
        validation_fn=validate,
        transform_fn=transform,
        default=default,
        error_msg=f"❌ Please choose a number (1-{len(value_label_pairs)}) or one of: {', '.join(valid_values)}")


def retry_or_exit(message: str, retry_prompt: str = "Retry? (y/n): ") -> bool:
    print(f"❌ {message}")
    logger.error(message)
    return ask_yes_no(retry_prompt, default="n")


@_wrap_special_commands
def ask_yes_no(prompt_text: str, default: str = "y") -> bool:
    result = prompt_with_validation(
        prompt_text=prompt_text,
        validation_fn=lambda x: x.lower() in ['y', 'yes', 'n', 'no'],
        transform_fn=lambda x: x.lower(),
        default=default,
        error_msg="❌ Please enter 'y' or 'n'."
    )
    return result in ['y', 'yes']


def print_section_header(title: str):
    """
    Print a section header with a consistent visual style.

    Args:
        title (str): The title to display in the header.
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


@_wrap_special_commands
def ask_nonempty_string(prompt_text: str, default: Optional[str] = None) -> str:
    """
    Prompt user for non-empty string input.

    Args:
        prompt_text (str): The prompt message to display.
        default (Optional[str]): Optional default value if user presses Enter.

    Returns:
        str: Validated non-empty input from the user.
    """
    return prompt_with_validation(
        prompt_text=prompt_text,
        validation_fn=lambda x: bool(x.strip()),
        default=default,
        error_msg="❌ Please enter a valid value."
    )


# Command-line argument handler for special commands
def handle_command_line_args(args=None):
    """
    Scan sys.argv or given args for special commands and execute them.
    Returns True if execution should stop before main program.
    """
    import sys
    argv = args if args is not None else sys.argv[1:]
    handled_any = False
    for arg in argv:
        if arg.startswith("--") and handle_command(arg):
            handled_any = True
    return handled_any


def display_storing_methods():
    print_section_header("📊 AVAILABLE VECTOR DATABASE STORING METHODS")

    descriptions = StoringMethod.get_descriptions()
    recommendations = StoringMethod.get_recommendations()

    for i, method in enumerate(StoringMethod.get_all_methods(), 1):
        print(f"\n{i}. {method.upper()}")
        print(f"   Description: {descriptions[method]}")
        print(f"   {recommendations[method]}")

    print("\n" + "=" * 60)


def get_user_storing_method(default_method: Optional[str] = None) -> str:
    display_storing_methods()
    methods = StoringMethod.get_all_methods()
    return ask_selection(
        prompt_text=f"\nSelect a storing method (1-{len(methods)}) or press Enter for default ({default_method}): ",
        options=methods,
        default=default_method
    )


import re
from urllib.parse import urlparse


def detect_source_type(source: str) -> str:
    """Detect the type of the input source."""
    if re.match(r"^\w+:\w+$", source):  # HuggingFace dataset with config
        return "huggingface_dataset_with_config"
    elif re.match(r"^\w+$", source):  # HuggingFace dataset without config
        return "huggingface_dataset"
    elif re.match(r"^https?://", source):  # URL
        parsed = urlparse(source)
        if parsed.netloc and parsed.path:
            return "url"
    elif os.path.exists(source):  # Local path
        return "local_path"
    return "invalid"


def get_user_source_path() -> str:
    print_section_header("📁 DATA SOURCE CONFIGURATION")
    print("Enter the path to your data source. Supported formats:")
    print("  • Local directory: /path/to/your/documents")
    print("  • URL: https://example.com/data.txt")
    print("  • HuggingFace dataset: squad:plain_text")
    print("  • HuggingFace dataset (no config): wikitext")
    print(f"\n🔁 Press Enter to use default as configured in config.INDEX_SOURCE_URL: {INDEX_SOURCE_URL}")

    return prompt_with_validation(
        prompt_text="🔤 Source path: ",
        validation_fn=lambda x: detect_source_type(x) != "invalid",
        default=INDEX_SOURCE_URL,
        transform_fn=lambda x: x.strip().strip('"').strip("'"),
        error_msg="❌ Invalid path format. Please try again."
    )


def confirm_configuration(storing_method: str, source_path: str, distance_function: Optional[str] = None) -> bool:
    print_section_header("⚙️  CONFIGURATION SUMMARY")
    print(f"Storing Method: {storing_method}")
    print(f"Source Path:    {source_path}")

    descriptions = StoringMethod.get_descriptions()
    if storing_method in descriptions:
        print(f"Description:    {descriptions[storing_method]}")

    if storing_method.lower() == "chroma" and distance_function:
        print(f"Distance Metric: {distance_function}")

    print("=" * 60)
    return ask_yes_no("Proceed with this configuration? (y/n): ", default="y")


def interactive_vector_db_setup() -> Tuple[str, str, Optional[str]]:
    """
    Complete interactive setup for vector database configuration.

    Returns:
        Tuple[str, str, Optional[str]]: (storing_method, source_path, distance_function)
    """
    print("\n🤖 VECTOR DATABASE SETUP WIZARD")
    print("Welcome! Let's configure your vector database.")

    try:
        storing_method = get_user_storing_method()
        source_path = get_user_source_path()
        distance_function = get_user_distance_function(storing_method)

        if confirm_configuration(storing_method, source_path, distance_function):
            logger.info(
                f"User selected configuration: method={storing_method}, source={source_path}, distance={distance_function}")
            return storing_method, source_path, distance_function
        else:
            print("\n🔄 Let's try again...")
            return interactive_vector_db_setup()

    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred during setup: {e}")
        logger.error(f"Error in interactive setup: {e}")
        sys.exit(1)


def quick_setup(method: Optional[str] = None, source: Optional[str] = None) -> Tuple[str, str]:
    """
    Quick setup with minimal prompts. Falls back to interactive if parameters missing.

    Args:
        method (Optional[str]): Storing method (if None, will prompt)
        source (Optional[str]): Source path (if None, will prompt)

    Returns:
        Tuple[str, str]: (storing_method, source_path)
    """
    # Validate and get method
    if method and StoringMethod.is_valid_method(method):
        storing_method = method
        print(f"✅ Using method: {storing_method}")
    else:
        if method:
            print(f"⚠️  Invalid method '{method}', please select a valid one:")
        storing_method = get_user_storing_method()

    # Get source path
    if source:
        source_path = source
        print(f"✅ Using source: {source_path}")
    else:
        source_path = get_user_source_path()

    return storing_method, source_path


# Convenience function for command line usage
def setup_from_args(args) -> Tuple[str, str]:
    """
    Setup configuration from command line arguments.

    Args:
        args: Argument parser object with 'method' and 'source' attributes

    Returns:
        Tuple[str, str]: (storing_method, source_path)
    """
    method = getattr(args, 'method', None)
    source = getattr(args, 'source', None)

    if method and source:
        # Both provided, validate and use
        if not StoringMethod.is_valid_method(method):
            retry_or_exit(f"Invalid storing method: {method}")

        print(f"✅ Using command line configuration:")
        print(f"   Method: {method}")
        print(f"   Source: {source}")
        return method, source
    else:
        # Missing parameters, use interactive setup
        print("⚠️  Missing configuration parameters, starting interactive setup...")
        return interactive_vector_db_setup()


def startup_initialization():
    """Initialize model and display startup information."""

    print("🚀 Loading model with MPS support...")
    with monitor_performance("startup_model_loading"):
        tokenizer, model = load_model()
        print(f"✅ Model loaded on device: {next(model.parameters()).device}")
        return tokenizer, model


def display_startup_banner():
    """Display the application startup banner."""
    print("\n" + "=" * 60)
    print("🤖 RAG SYSTEM - Enhanced MPS Support")
    print("=" * 60)
    device = get_optimal_device()
    if device.type == "mps":
        print("🍎 Apple Silicon GPU Detected - MPS Enabled")
    elif device.type == "cuda":
        print("🔥 NVIDIA GPU Detected - CUDA Enabled")
    else:
        print("💻 CPU Mode - No GPU Acceleration")
    print("=" * 60)
    return device


def get_user_distance_function(storing_method: str) -> Optional[str]:
    if storing_method.lower() != "chroma":
        return None

    print("\n📏 Select distance function for similarity search:")
    print("1. Cosine")
    print("2. Euclidean (L2)")
    print("3. Inner Product")

    choice = input("Enter your choice [1-3]: ").strip()

    mapping = {
        "1": "cosine",
        "2": "l2",
        "3": "ip"
    }

    return mapping.get(choice, "cosine")  # fallback to cosine


from utility.distance_metrics import DistanceMetric


def setup_vector_database():
    """Setup vector database with user interaction and dataset/corpus selection."""

    print("\n📚 VECTOR DATABASE SETUP")
    print("-" * 30)

    # Step 1: Dataset selection and corpus association
    dataset_key, corpus_source = select_dataset_and_corpus()

    # If user chose manual mode (no dataset), skip corpus and go to vector DB selection
    if dataset_key is None:
        print("\n⚠️ No dataset selected. Skipping dataset/corpus loading.")
        # Fallback to interactive or no-DB mode
        print("(Default: y - use Vector DB)")
        use_vector_db = ask_yes_no("Do you want to use vector database for context retrieval? (y/n): ", default='y')
        if not use_vector_db:
            print("📝 Running in simple Q&A mode (no context retrieval)")
            return None, None, None, None, None
        try:
            storing_method, source_path, distance_function = interactive_vector_db_setup()
            if distance_function:
                try:
                    distance_metric = DistanceMetric(distance_function.lower())
                except ValueError:
                    print(f"⚠️ Unsupported distance metric '{distance_function}', falling back to COSINE")
                    distance_metric = DistanceMetric.COSINE
            else:
                distance_metric = DistanceMetric.COSINE
            with monitor_performance("vector_db_loading"):
                logger.info(
                    f"Loading Vector DB with method: {storing_method}, source: {source_path}, distance: {distance_metric}")
                vector_db, embedding_model = load_vector_db(
                    source_path=source_path,
                    storing_method=storing_method,
                    distance_metric=distance_metric
                )
                print("✅ Vector DB loaded successfully")
                stats = vector_db.get_stats()
                print(f"\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"  • {key}: {value}")
                from configurations.config import HF_MODEL_NAME
                print("HF_MODEL_NAME:", HF_MODEL_NAME)
                emb = embedding_model.get_text_embedding("test")
                print("Embedding dimension:", len(emb))
                print("Vector DB stats:", vector_db.get_stats())
                return vector_db, embedding_model, storing_method, source_path, distance_metric
        except Exception as e:
            logger.error(f"Failed to load vector DB: {e}")
            print(f"❌ Vector DB setup failed: {e}")
            print("(Default: y - retry with Vector DB)")
            retry = ask_yes_no("Retry with Vector DB? (y/n): ", default='y')
            if not retry:
                print("⚠️  Continuing without vector database...")
                return None, None, None, None, None
            else:
                # Retry the same setup function to re-download/rebuild
                return setup_vector_database()

    # If the dataset defines a corpus, skip vector DB prompts and use it
    if corpus_source:
        # Use default storing method and distance metric if desired, or prompt if needed
        from configurations.config import DEFAULT_STORING_METHOD, DEFAULT_DISTANCE_METRIC
        if DEFAULT_STORING_METHOD:
            storing_method = DEFAULT_STORING_METHOD
            print(f"✅ Using storing method from config: {storing_method}")
        else:
            print("⚙️ No storing method defined in config. Using default: chroma")
            storing_method = "chroma"

        if DEFAULT_DISTANCE_METRIC:
            try:
                distance_metric = DistanceMetric(DEFAULT_DISTANCE_METRIC.lower())
                print(f"✅ Using distance metric from config: {DEFAULT_DISTANCE_METRIC}")
            except ValueError:
                print(f"⚠️ Invalid distance metric '{DEFAULT_DISTANCE_METRIC}' in config. Falling back to COSINE.")
                distance_metric = DistanceMetric.COSINE
        else:
            print("⚙️ No distance metric defined in config. Using default: COSINE")
            distance_metric = DistanceMetric.COSINE
        source_path = corpus_source
        print(f"\n🔄 Automatically loading vector DB for dataset '{dataset_key}' using corpus '{corpus_source}'...")
        try:
            with monitor_performance("vector_db_loading"):
                from utility.embedding_utils import get_text_embedding
                logger.info(
                    f"Loading Vector DB with method: {storing_method}, source: {source_path}, distance: {distance_metric}")
                vector_db, embedding_model = load_vector_db(
                    source_path=source_path,
                    storing_method=storing_method,
                    distance_metric=distance_metric
                )
                print("✅ Vector DB loaded successfully")
                stats = vector_db.get_stats()
                print(f"\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"  • {key}: {value}")
                from configurations.config import HF_MODEL_NAME
                print("HF_MODEL_NAME:", HF_MODEL_NAME)
                emb = embedding_model.get_text_embedding("test")
                print("Embedding dimension:", len(emb))
                print("Vector DB stats:", vector_db.get_stats())
                return vector_db, embedding_model, storing_method, source_path, distance_metric
        except Exception as e:
            logger.error(f"Failed to load vector DB: {e}")
            print(f"❌ Vector DB setup failed: {e}")
            if "No files found" in str(e) and corpus_source:
                print(f"🌐 Corpus '{corpus_source}' not found locally. Attempting to rebuild via load_vector_db...")
                try:
                    vector_db, embedding_model = load_vector_db(
                        source_path=corpus_source,
                        storing_method=storing_method,
                        distance_metric=distance_metric
                    )
                    print("✅ Vector DB successfully built or downloaded by load_vector_db().")
                    stats = vector_db.get_stats()
                    print(f"📊 Database Statistics:")
                    for key, value in stats.items():
                        print(f"  • {key}: {value}")
                    return vector_db, embedding_model, storing_method, source_path, distance_metric
                except Exception as rebuild_e:
                    print(f"❌ Rebuild via load_vector_db() failed for corpus '{corpus_source}': {rebuild_e}")
                    print("⚠️  Continuing without vector database...")
                    return None, None, None, None, None
            else:
                print("(Default: y - retry with Vector DB)")
                retry = ask_yes_no("Retry with Vector DB? (y/n): ", default='y')
                if not retry:
                    print("⚠️  Continuing without vector database...")
                    return None, None, None, None, None
                else:
                    # Retry the same setup function to re-download/rebuild
                    return setup_vector_database()
    else:
        # No corpus defined: fall back to legacy interactive vector DB setup
        print("(Default: y - use Vector DB)")
        use_vector_db = ask_yes_no("Do you want to use vector database for context retrieval? (y/n): ", default='y')
        if not use_vector_db:
            print("📝 Running in simple Q&A mode (no context retrieval)")
            return None, None, None, None, None
        try:
            storing_method, source_path, distance_function = interactive_vector_db_setup()
            if distance_function:
                try:
                    distance_metric = DistanceMetric(distance_function.lower())
                except ValueError:
                    print(f"⚠️ Unsupported distance metric '{distance_function}', falling back to COSINE")
                    distance_metric = DistanceMetric.COSINE
            else:
                distance_metric = DistanceMetric.COSINE
            with monitor_performance("vector_db_loading"):
                logger.info(
                    f"Loading Vector DB with method: {storing_method}, source: {source_path}, distance: {distance_metric}")
                vector_db, embedding_model = load_vector_db(
                    source_path=source_path,
                    storing_method=storing_method,
                    distance_metric=distance_metric
                )
                print("✅ Vector DB loaded successfully")
                stats = vector_db.get_stats()
                print(f"\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"  • {key}: {value}")
                from configurations.config import HF_MODEL_NAME
                print("HF_MODEL_NAME:", HF_MODEL_NAME)
                emb = embedding_model.get_text_embedding("test")
                print("Embedding dimension:", len(emb))
                print("Vector DB stats:", vector_db.get_stats())
                return vector_db, embedding_model, storing_method, source_path, distance_metric
        except Exception as e:
            logger.error(f"Failed to load vector DB: {e}")
            print(f"❌ Vector DB setup failed: {e}")
            print("(Default: y - retry with Vector DB)")
            retry = ask_yes_no("Retry with Vector DB? (y/n): ", default='y')
            if not retry:
                print("⚠️  Continuing without vector database...")
                return None, None, None, None, None
            else:
                # Retry the same setup function to re-download/rebuild
                return setup_vector_database()


def validate_prerequisites(mode: str, dataset_key, vector_db) -> bool:
    """
    Validate that the required components are available before running a mode.
    """
    if mode == "evaluation" and (dataset_key is None or vector_db is None):
        print("❌ Evaluation mode requires both a dataset and a vector database.")
        return False

    if mode == "retrieval_base" and (dataset_key is None or vector_db is None):
        print("❌ Retrieval Base experiment requires both a dataset and a vector database.")
        return False

    if mode == "noise" and vector_db is None:
        print("❌ Noise robustness test requires a vector database.")
        return False

    if mode == "development" and vector_db is None:
        print("⚠️ Development mode may have limited functionality without a vector DB.")

    return True


def display_main_menu(vector_db=None, dataset_key=None):
    """Display the main menu and get user choice."""
    print_section_header("🎯 SELECT MODE")
    eval_label = "📊 Evaluation Mode" if dataset_key and vector_db else "📊 Evaluation Mode (disabled – no dataset/DB)"
    dev_label = "🔧 Development Test Mode" if vector_db else "🔧 Development Test Mode (limited – no DB)"

    options = [
        ("1", "💬 Interactive Q&A Mode"),
        ("2", eval_label),
        ("3", dev_label),
        ("4", "📈 Results Analysis"),
        ("5", "⚙️ System Information"),
        ("6", "📥️ Download QA Dataset"),
        ("7", "🧪 Experiments & Evaluation"),
        ("8", "🚪 Exit"),
        ("9", "🔄 Reset Vector DB & Embedding Model")
    ]

    for key, label in options:
        print(f"{key}. {label}")

    return ask_selection(
        prompt_text="\nEnter your choice: ",
        options=[(key, key) for key, _ in options]
    )


def show_experiments_menu():
    """
    Display the experiments and evaluation menu and get user choice.
    """
    print_section_header("🧪 EXPERIMENTS & EVALUATION")
    options = [
        ("1", "🧪 Simple Retrieval Evaluation"),
        ("2", "🔬 Noise Robustness Experiment"),
        ("3", "🔧 Development Test Mode"),
        ("4", "🧪 New Experiment (coming soon)"),
        ("5", "⬅️ Back to Main Menu")
    ]
    for key, label in options:
        print(f"{key}. {label}")
    return ask_selection(
        prompt_text="\nSelect experiment: ",
        options=[(key, key) for key, _ in options]
    )


def flush_input():
    """Flush accidental keyboard input from the terminal buffer."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # Normal in some environments


def show_system_info(vector_db, model, device):
    """Display comprehensive system information."""
    print("\n⚙️  SYSTEM INFORMATION")
    print("=" * 25)

    # Device information
    profile_gpu(model, device)

    # Model information
    print(f"\n🤖 Model Information:")
    try:
        info = get_embedding_model_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
    except Exception as e:
        print(f"  ❌ Could not get model info: {e}")

    # Vector DB information
    if vector_db:
        print(f"\n📚 Vector Database:")
        stats = vector_db.get_stats()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    else:
        print(f"\n📚 Vector Database: Not loaded")

    # Performance stats
    if PERFORMANCE_AVAILABLE:
        print(f"\n📊 Performance Statistics:")
        performance_report()

    # Cache stats
    if CACHE_AVAILABLE:
        print(f"\n💾 Cache Statistics:")
        cache_stats()


@track_performance("gpu_profiling")
def profile_gpu(model, device):
    """Enhanced GPU profiling with MPS support."""

    print(f"\n🖥️  DEVICE INFORMATION")
    print("-" * 25)

    if device.type == "mps":
        print("🍎 MPS (Apple Silicon) GPU detected:")
        print("  • Memory monitoring is limited on MPS")
        print("  • Using float32 precision for compatibility")
        try:
            print(f"  • Model device: {next(model.parameters()).device}")
            print(f"  • Model dtype: {next(model.parameters()).dtype}")

            bfloat16_count = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
            if bfloat16_count > 0:
                print(f"  ⚠️  Warning: {bfloat16_count} bfloat16 parameters detected!")
            else:
                print("  ✅ All parameters are MPS-compatible")

        except Exception as e:
            print(f"  • Could not get model info: {e}")

    elif device.type == "cuda":
        print("🔥 CUDA GPU Utilization:")
        print(torch.cuda.memory_summary(device="cuda"))
    else:
        print("💻 CPU Mode - No GPU acceleration")


@track_performance("interactive_mode")
def run_interactive_mode(vector_db, embedding_model, tokenizer, model, device):
    """Run the interactive Q&A mode.
    This mode allows users to ask questions and get answers win a loop with optional context retrieval.

    Args:
        ""vector_db (Optional[object]): Vector database instance for context retrieval.
        embedding_model (Optional[object]): Embedding model for vectorization.
        tokenizer (Optional[object]): Tokenizer for the model.
        model (Optional[object]): Language model for generating answers.
        device (torch.device): Device to run the model on (CPU/GPU/MPS).
        Returns:
            "None

    Parameters
    ----------
    device
    model
    tokenizer
    embedding_model
    vector_db
    """

    print("\n💬 INTERACTIVE Q&A MODE")
    print("=" * 30)

    if vector_db is None:
        print("📝 Running in simple Q&A mode (no context retrieval)")
    else:
        print("📚 Running with context retrieval enabled")
        print(f"🗃️  Database type: {vector_db.db_type}")

    print("💡 Type 'exit', 'quit', or 'q' to return to main menu")
    print("💡 Type 'help' for available commands\n")

    while True:
        flush_input()
        user_prompt = input("🤔 Your question: ").strip()

        if user_prompt.lower() in ['exit', 'quit', 'q']:
            print("🔙 Returning to main menu...")
            break

        if user_prompt.lower() == 'help':
            print("\n📖 Available Commands:")
            print("  • exit/quit/q - Return to main menu")
            print("  • help - Show this help")
            print("  • stats - Show database statistics")
            print("  • device - Show device information")
            continue

        if user_prompt.lower() == 'stats' and vector_db:
            stats = vector_db.get_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                print(f"  • {key}: {value}")
            continue

        if user_prompt.lower() == 'device':
            profile_gpu(model, device)
            continue

        if not user_prompt:
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


@track_performance("evaluation_mode")
def run_evaluation_mode(vector_db, embedding_model, tokenizer, model, device):
    """
    Run the evaluation mode using a QA dataset and evaluate the model's performance.

    Args:
        vector_db: Vector DB instance for context retrieval.
        embedding_model: Embedding model for vectorization.
        tokenizer: Tokenizer for the LLM.
        model: The LLM model used for answering.
        device: Device on which model is running.
    """
    if not validate_prerequisites("evaluation", ACTIVE_QA_DATASET, vector_db):
        return

    print("\n📊 EVALUATION MODE")
    print("=" * 25)

    mode_choice = ask_selection(
        prompt_text="Select mode: \n- (e)numeration \n- (t)rilateration retriever  \n- (h)ill climbing: ",
        options=["e", "t", "h"],
        default="e"
    )

    results_logger = ResultsLogger(top_k=RETRIEVER_TOP_K, mode="hill" if mode_choice == "h" else "enum")

    try:
        queries = load_qa_queries(NQ_SAMPLE_SIZE)
    except Exception as e:
        print(f"❌ Failed to load QA dataset: {e}")
        return

    print(f"📁 Processing {len(queries)} queries from dataset '{ACTIVE_QA_DATASET}'...")
    with monitor_performance("evaluation_mode_execution"):
        for i, query in enumerate(queries):
            print(f"🔍 Processing query {i + 1}/{len(queries)}: {query[:80]}...")

            with monitor_performance(f"query_processing_{i + 1}"):
                try:
                    if mode_choice == "h":
                        from scripts.evaluator import hill_climb_documents
                        result = hill_climb_documents(
                            i=i, num=len(queries), query=query, index=vector_db,
                            llm_model=model, tokenizer=tokenizer,
                            embedding_model=embedding_model, top_k=RETRIEVER_TOP_K,
                            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
                        )
                    elif mode_choice == "t":
                        from vector_db.trilateration_retriever import TrilaterationRetriever
                        retriever = TrilaterationRetriever(
                            embedding_model,
                            vector_db,
                            iterative=TRILATERATION_ITERATIVE,
                            max_refine_steps=TRILATERATION_MAX_REFINES,
                            convergence_tol=TRILATERATION_CONVERGENCE_TOL
                        )
                        result = retriever.retrieve(query)
                    else:
                        result = enumerate_top_documents(
                            i=i, num=len(queries), query=query, index=vector_db,
                            embedding_model=embedding_model,
                            top_k=RETRIEVER_TOP_K, convert_to_vector=False
                        )
                    log_message = (f"✅ Query {i + 1} processed. Answer: {result.get('query', 'N/A')[:100]}, "
                                   f"with distance of : {result.get('distance', 'N/A')}...")
                    results_logger.log(log_message)

                except Exception as e:
                    logger.error(f"❌ Error in evaluation query {i + 1}: {e}")
                    print(f"❌ Error processing query {i + 1}: {e}")

    print("✅ Evaluation completed.")


@track_performance("development_test")
def run_development_test(vector_db, embedding_model, tokenizer, model, device):
    """Enhanced development test with MPS compatibility."""
    validate_prerequisites("development", ACTIVE_QA_DATASET, vector_db)
    print("\n🔧 DEVELOPMENT TEST MODE")
    print("=" * 30)

    # System info
    profile_gpu(model, device)

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
    try:
        from modules.model_loader import get_model_capabilities
        capabilities = get_model_capabilities(model)
        for key, value in capabilities.items():
            status = "✅" if value else "⚠️" if key == 'is_causal_lm' else "ℹ️"
            print(f"  {status} {key}: {value}")
    except Exception as e:
        print(f"❌ Model capabilities test failed: {e}")

    # Vector DB test
    print("\n📚 Testing vector database...")
    if vector_db:
        try:
            stats = vector_db.get_stats()
            print(f"✅ Vector DB loaded: {stats['db_type']}")
            print(f"  Statistics: {stats}")
        except Exception as e:
            print(f"❌ Vector DB stats failed: {e}")
    else:
        print("⚠️  No vector database loaded")

    # Query test
    print("\n💬 Testing query processing...")
    test_query = "What is artificial intelligence?"
    try:
        device = get_optimal_device()
        with monitor_performance("dev_query_test"):
            result = process_query_with_context(
                test_query, model, tokenizer, device, vector_db, embedding_model
            )

        print(f"✅ Test query result: {result['answer'][:100]}...")
        if 'score' in result:
            print(f"📊 Score: {result['score']}")
        print(f"🖥️  Device used: {result.get('device_used', 'unknown')}")
    except Exception as e:
        print(f"❌ Query test failed: {e}")

    print("\n🎉 Development test completed")


def run_noise_robustness_experiment(
        vector_db: VectorDBInterface,
        embedding_model: HuggingFaceEmbedding):
    if not validate_prerequisites("noise", ACTIVE_QA_DATASET, vector_db):
        return
    db_type = vector_db.db_type
    distance_metric = vector_db.distance_metric
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    default_filename = f"{db_type}_{distance_metric}_{timestamp}.xlsx"

    filename = prompt_with_validation(
        f"Enter output Excel filename (press Enter to use default: {default_filename}):\n",
        lambda s: s == "" or s.endswith('.xlsx'),
        default=default_filename
    )

    if not filename or not filename.endswith('.xlsx'):
        logger.warning("Invalid or empty filename provided. Using default name.")
        filename = default_filename

    output_dir = os.path.join(PROJECT_PATH, "results", "noise_robustness")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    run_noise_experiment(
        vector_db=vector_db,
        embed_model=embedding_model,
        top_k=RETRIEVER_TOP_K,
        output_path=output_path
    )


def run_retrieval_base_algorithm_experiment(
        vector_db: VectorDBInterface,
        embedding_model: HuggingFaceEmbedding,
        evaluator_model: torch.nn.Module
):
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


############################################################
# User-facing message and prompt helpers (centralized I/O) #
############################################################

def show_exit_message():
    """Display a goodbye message."""
    print("\n👋 Goodbye! Have a great day!")


def confirm_use_vector_db() -> bool:
    """Ask user if they want to use a vector database."""
    return ask_yes_no("Do you want to use vector database for context retrieval? (y/n): ", default='y')


def show_vector_db_success():
    """Display a success message after loading the vector database."""
    print("✅ Vector DB loaded successfully")


def show_error_message(message: str):
    """Display a formatted error message."""
    print(f"❌ {message}")


def show_goodbye_message():
    """Alias for exit message."""
    show_exit_message()


def confirm_reset_vector_db() -> bool:
    """Ask user to confirm vector DB reset."""
    return ask_yes_no("Are you sure you want to reset the Vector DB and Embedding Model? (y/n): ", default='n')


def show_mode_choice_banner():
    """Display a banner for selecting execution mode."""
    print("\n📊 SELECT EXECUTION MODE")
    print("-" * 40)
    print("1. Enumeration")
    print("2. Hill Climbing")
    print("3. Trilateration")
    print("-" * 40)


def show_evaluation_completed():
    """Display completion message for evaluation."""
    print("✅ Evaluation completed successfully.")


def show_performance_summary_notice():
    """Display performance summary location notice."""
    print("\n🧾 Performance summary saved to: results/performance_metrics.json")
    print("📊 You can view it later or print it now.")
