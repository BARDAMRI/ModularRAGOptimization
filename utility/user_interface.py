"""
User interface utilities for selecting vector database configurations
"""
import os
import sys
import termios
from enum import Enum
from typing import Optional, Tuple, Callable
import torch

from configurations.config import NQ_SAMPLE_SIZE, MAX_NEW_TOKENS, TEMPERATURE, QUALITY_THRESHOLD, MAX_RETRIES, \
    QA_DATASET_NAME, INDEX_SOURCE_URL
from matrics.results_logger import ResultsLogger
from modules.model_loader import load_model
from modules.query import process_query_with_context
from scripts.evaluator import enumerate_top_documents
from scripts.qa_data_set_loader import load_qa_queries
from utility.device_utils import get_optimal_device
from utility.logger import logger
from vector_db.indexer import get_embedding_model_info, load_vector_db
from vector_db.storing_methods import StoringMethod

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

        # Remove surrounding quotes if present
        if user_input.startswith('"') and user_input.endswith('"'):
            user_input = user_input[1:-1].strip()
        elif user_input.startswith("'") and user_input.endswith("'"):
            user_input = user_input[1:-1].strip()

        if not user_input and default is not None:
            print(f"✅ Using default: {default}")
            return default

        try:
            normalized_input = user_input.lower()
            if validation_fn(normalized_input):
                result = transform_fn(normalized_input) if transform_fn else normalized_input
                print(f"✅ Selected: {result}")
                return result
        except Exception:
            pass

        print(error_msg)


def ask_selection(prompt_text: str, options: list, default: Optional[str] = None) -> str:
    # Create mapping from numbers (1-based) to options
    index_to_option = {str(i + 1): option for i, option in enumerate(options)}
    valid_inputs = set(options) | set(index_to_option.keys())

    return prompt_with_validation(
        prompt_text=prompt_text + ''.join([f"\n  {i + 1}. {option}" for i, option in enumerate(options)]),
        validation_fn=lambda x: x in valid_inputs,
        default=default,
        transform_fn=lambda x: index_to_option.get(x, x),  # If number, map to option; else return as-is
        error_msg=f"❌ Please choose a number (1-{len(options)}) or one of: {', '.join(options)}"
    )


def retry_or_exit(message: str, retry_prompt: str = "Retry? (y/n): ") -> bool:
    print(f"❌ {message}")
    logger.error(message)
    return ask_yes_no(retry_prompt, default="n")


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


def confirm_configuration(storing_method: str, source_path: str) -> bool:
    print_section_header("⚙️  CONFIGURATION SUMMARY")
    print(f"Storing Method: {storing_method}")
    print(f"Source Path:    {source_path}")

    descriptions = StoringMethod.get_descriptions()
    if storing_method in descriptions:
        print(f"Description:    {descriptions[storing_method]}")
    print("=" * 60)

    return ask_yes_no("Proceed with this configuration? (y/n): ", default="y")


def interactive_vector_db_setup() -> Tuple[str, str]:
    """
    Complete interactive setup for vector database configuration.

    Returns:
        Tuple[str, str]: (storing_method, source_path)
    """
    print("\n🤖 VECTOR DATABASE SETUP WIZARD")
    print("Welcome! Let's configure your vector database.")

    try:
        # Get storing method
        storing_method = get_user_storing_method()

        # Get source path
        source_path = get_user_source_path()

        # Confirm configuration
        if confirm_configuration(storing_method, source_path):
            logger.info(f"User selected configuration: method={storing_method}, source={source_path}")
            return storing_method, source_path
        else:
            print("\n🔄 Let's try again...")
            return interactive_vector_db_setup()  # Recursive call to restart

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


def setup_vector_database() -> Tuple[Optional[object], Optional[object]]:
    """Setup vector database with user interaction."""

    print("\n📚 VECTOR DATABASE SETUP")
    print("-" * 30)

    use_vector_db = ask_yes_no("Do you want to use vector database for context retrieval? (y/n): ", default='n')

    if not use_vector_db:
        print("📝 Running in simple Q&A mode (no context retrieval)")
        return None, None

    try:
        storing_method, source_path = interactive_vector_db_setup()

        with monitor_performance("vector_db_loading"):
            logger.info(f"Loading Vector DB with method: {storing_method}, source: {source_path}")
            vector_db, embedding_model = load_vector_db(
                source_path=source_path,
                storing_method=storing_method
            )

            print("✅ Vector DB loaded successfully")

            stats = vector_db.get_stats()
            print(f"\n📊 Database Statistics:")
            for key, value in stats.items():
                print(f"  • {key}: {value}")

            return vector_db, embedding_model

    except Exception as e:
        logger.error(f"Failed to load vector DB: {e}")
        print(f"❌ Vector DB setup failed: {e}")

        retry = ask_yes_no("Continue without vector DB? (y/n): ", default='y')
        if retry:
            print("⚠️  Continuing without vector database...")
            return None, None
        else:
            sys.exit(1)


def display_main_menu():
    """Display the main menu and get user choice."""
    print("\n🎯 SELECT MODE")
    print("-" * 30)

    return ask_selection(
        prompt_text="Enter your choice:",
        options=[
            ("1", "💬 Interactive Q&A Mode"),
            ("2", "📊 Evaluation Mode (Natural Questions)"),
            ("3", "🔧 Development Test Mode"),
            ("4", "📈 Results Analysis"),
            ("5", "⚙️ System Information"),
            ("6", "📥️ Download QA Dataset"),
            ("7", "🚪 Exit")
        ]
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
    if vector_db is None:
        print("❌ Evaluation mode requires Vector DB. Please restart and set up vector database.")
        return

    print("\n📊 EVALUATION MODE")
    print("=" * 25)

    mode_choice = ask_selection(
        prompt_text="Select mode: (e)numeration / (h)ill climbing: ",
        options=["e", "h"],
        default="e"
    )

    results_logger = ResultsLogger(top_k=5, mode="hill" if mode_choice == "h" else "enum")

    try:
        queries = load_qa_queries(NQ_SAMPLE_SIZE)
    except Exception as e:
        print(f"❌ Failed to load QA dataset: {e}")
        return

    print(f"📁 Processing {len(queries)} queries from dataset '{QA_DATASET_NAME}'...")

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
                            embedding_model=embedding_model, top_k=5,
                            max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
                        )
                    else:
                        result = enumerate_top_documents(
                            i=i, num=len(queries), query=query, index=vector_db,
                            embedding_model=embedding_model,
                            top_k=5, convert_to_vector=False
                        )
                    results_logger.log(result)

                except Exception as e:
                    logger.error(f"❌ Error in evaluation query {i + 1}: {e}")
                    print(f"❌ Error processing query {i + 1}: {e}")

    print("✅ Evaluation completed.")


@track_performance("development_test")
def run_development_test(vector_db, embedding_model, tokenizer, model, device):
    """Enhanced development test with MPS compatibility."""
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
