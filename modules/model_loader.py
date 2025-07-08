# modules/model_loader.py - Complete version with performance monitoring
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
import torch
from configurations.config import MODEL_PATH, FORCE_CPU, OPTIMIZE_FOR_MPS, USE_MIXED_PRECISION
from typing import Tuple
from utility.logger import logger
try:
    from utility.performance import monitor_performance, track_performance

    PERFORMANCE_AVAILABLE = True
except ImportError:
    logger.warning("Performance monitoring not available. Install with: pip install psutil")
    PERFORMANCE_AVAILABLE = False

    # Create dummy decorators if performance module not available
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


def get_optimal_device():
    """
    Determine the best available device for model execution.

    Checks device availability in priority order: FORCE_CPU config -> MPS -> CUDA -> CPU fallback.

    Returns:
        torch.device: The optimal device object (cpu, mps, or cuda)
    """
    with monitor_performance("device_detection"):
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


@track_performance("complete_model_loading")
def load_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load and optimize a language model with GPU acceleration support and performance monitoring.

    Automatically detects the best available device (MPS/CUDA/CPU) and applies
    device-specific optimizations for maximum performance and compatibility.

    Returns:
        Tuple[AutoTokenizer, torch.nn.Module]: A tuple containing:
            - tokenizer: The loaded tokenizer with padding token configured
            - model: The loaded and optimized model on the appropriate device

    Raises:
        Exception: If model loading fails on all attempted methods
    """
    logger.info(f"Loading Model {MODEL_PATH}...")

    # Get optimal device with performance tracking
    device = get_optimal_device()

    # Load tokenizer with performance tracking
    with monitor_performance("tokenizer_loading"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Added padding token to tokenizer.")
        logger.info("Tokenizer loaded successfully.")

    # Try to load as CausalLM first (for text generation)
    try:
        with monitor_performance("model_download_and_instantiation"):
            # Device-specific model loading
            if device.type == "mps":
                # MPS-optimized loading
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.float32,  # MPS requires float32
                    low_cpu_mem_usage=True,
                    device_map=None  # Don't use device_map with MPS
                )
                logger.info("Loaded model for MPS (Apple Silicon)")

            elif device.type == "cuda":
                # CUDA-optimized loading
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
                logger.info("Loaded model for CPU")

        # Move model to device with performance tracking
        with monitor_performance("model_to_device"):
            if device.type != "cuda" or not hasattr(model, 'hf_device_map'):
                # Only move manually if not using device_map
                model = model.to(device)
                logger.info(f"Model moved to {device}")

        # Apply optimizations with performance tracking
        with monitor_performance("model_optimization"):
            # Skip torch.compile for MPS as it can cause issues
            if (torch.__version__.startswith("2") and
                    device.type != "mps" and
                    not getattr(torch.backends, 'mps_compile_disabled', False)):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model optimized with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            elif device.type == "mps":
                logger.info("Skipping torch.compile (MPS compatibility)")
            else:
                logger.info("torch.compile skipped")

        logger.info("Loaded AutoModelForCausalLM (supports text generation)")
        return tokenizer, model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")

        # Fallback attempt with performance tracking
        logger.info("Attempting fallback loading...")
        try:
            with monitor_performance("fallback_model_loading"):
                model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
                model = model.to(device)
            logger.info("Fallback loading successful")
            return tokenizer, model
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            raise e2


@track_performance("model_capabilities_analysis")
def get_model_capabilities(model) -> dict:
    """
    Analyze and report the capabilities and configuration of a loaded model.

    Args:
        model: The loaded PyTorch model to analyze

    Returns:
        dict: Dictionary containing model information:
            - can_generate (bool): Whether model supports text generation
            - model_type (str): Class name of the model
            - is_causal_lm (bool): Whether it's a causal language model
            - device (str): Device the model is currently on
            - dtype (str): Data type of model parameters
            - parameter_count (int): Total number of model parameters
    """
    with monitor_performance("capability_analysis"):
        # Get the first parameter to extract device and dtype info
        try:
            first_param = next(iter(model.parameters()))
            device = str(first_param.device)
            dtype = str(first_param.dtype)
        except StopIteration:
            # Handle case where model has no parameters
            device = "unknown"
            dtype = "unknown"

        # Count total parameters
        parameter_count = sum(p.numel() for p in model.parameters())

        capabilities = {
            'can_generate': hasattr(model, 'generate'),
            'model_type': type(model).__name__,
            'is_causal_lm': 'CausalLM' in type(model).__name__,
            'device': device,
            'dtype': dtype,
            'parameter_count': parameter_count,
        }

        return capabilities


@track_performance("gpu_memory_monitoring")
def monitor_gpu_memory():
    """
    Monitor and log current GPU memory usage.

    Provides device-specific memory information:
    - MPS: Limited monitoring capabilities (Apple Silicon limitation)
    - CUDA: Detailed memory allocation and reservation info
    - CPU: No GPU memory to monitor

    Logs memory usage information at INFO level.
    """
    if torch.backends.mps.is_available():
        # MPS doesn't have detailed memory reporting yet
        logger.info("MPS: Memory monitoring limited on Apple Silicon")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(f"CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        logger.info("CPU: No GPU memory to monitor")


@track_performance("complete_model_test")
def test_model_loading():
    """
    Comprehensive test function to verify GPU-optimized model loading and generation.

    Performs the following tests:
    1. Loads model and tokenizer with GPU optimization
    2. Analyzes and reports model capabilities
    3. Monitors memory usage before and after operations
    4. Tests actual text generation on the selected device
    5. Validates that text generation works correctly

    Returns:
        bool: True if model loading and text generation succeed, False otherwise

    Logs detailed information about each test step and any encountered issues.
    """
    logger.info("Testing GPU-optimized model loading...")

    try:
        # Load model and tokenizer
        with monitor_performance("test_model_loading"):
            tokenizer, model = load_model()

        # Analyze capabilities
        with monitor_performance("test_capabilities_analysis"):
            capabilities = get_model_capabilities(model)

        logger.info("Model loading test results:")
        for key, value in capabilities.items():
            if key == 'can_generate':
                status = "PASS" if value else "FAIL"
            elif key == 'is_causal_lm':
                status = "PASS" if value else "WARN"
            else:
                status = "INFO"
            logger.info(f"  {status} {key}: {value}")

        # Monitor memory after loading
        monitor_gpu_memory()

        if capabilities['can_generate']:
            logger.info("Model supports text generation")

            # Quick generation test with performance tracking
            with monitor_performance("test_text_generation"):
                device = next(model.parameters()).device
                test_input = tokenizer("Hello", return_tensors="pt")

                # Move inputs to same device as model
                test_input = {k: v.to(device) for k, v in test_input.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **test_input,
                        max_new_tokens=5,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )

                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Test generation on {device}: '{result}'")

            # Check memory after generation
            monitor_gpu_memory()
        else:
            logger.warning("Model does NOT support text generation")

        return capabilities['can_generate']

    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


def benchmark_model_performance(num_iterations: int = 5):
    """
    Benchmark model performance across multiple iterations.

    Args:
        num_iterations (int): Number of iterations to run for benchmarking

    Returns:
        dict: Benchmark results with timing statistics
    """
    logger.info(f"Starting model performance benchmark ({num_iterations} iterations)")

    results = {
        'loading_times': [],
        'generation_times': [],
        'total_times': []
    }

    for i in range(num_iterations):
        logger.info(f"Iteration {i + 1}/{num_iterations}")

        # Clear any cached models (if applicable)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Time complete loading process
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start_time:
            start_time.record()
        else:
            import time
            start_cpu = time.time()

        # Load model
        with monitor_performance(f"benchmark_iteration_{i + 1}"):
            tokenizer, model = load_model()

            # Quick generation test
            device = next(model.parameters()).device
            test_input = tokenizer("Hello world", return_tensors="pt")
            test_input = {k: v.to(device) for k, v in test_input.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **test_input,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.eos_token_id
                )

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            iteration_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            iteration_time = time.time() - start_cpu

        results['total_times'].append(iteration_time)
        logger.info(f"  Iteration {i + 1} completed in {iteration_time:.2f}s")

    # Calculate statistics
    total_times = results['total_times']
    benchmark_stats = {
        'iterations': num_iterations,
        'avg_time': sum(total_times) / len(total_times),
        'min_time': min(total_times),
        'max_time': max(total_times),
        'total_time': sum(total_times)
    }

    logger.info("Benchmark Results:")
    logger.info(f"  Average time: {benchmark_stats['avg_time']:.2f}s")
    logger.info(f"  Min time: {benchmark_stats['min_time']:.2f}s")
    logger.info(f"  Max time: {benchmark_stats['max_time']:.2f}s")
    logger.info(f"  Total time: {benchmark_stats['total_time']:.2f}s")

    return benchmark_stats


def get_device_info():
    """
    Get comprehensive device information for debugging and optimization.

    Returns:
        dict: Device information including capabilities and memory
    """
    device_info = {
        'cpu_count': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'optimal_device': str(get_optimal_device())
    }

    if torch.cuda.is_available():
        device_info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            'cuda_compute_capability': torch.cuda.get_device_capability(0)
        })

    if torch.backends.mps.is_available():
        device_info.update({
            'mps_device': 'Apple Silicon GPU detected'
        })

    return device_info


def print_system_info():
    """Print comprehensive system information for debugging."""
    logger.info("System Information:")

    device_info = get_device_info()
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Model path: {MODEL_PATH}")
    logger.info(f"  Force CPU: {FORCE_CPU}")
    logger.info(f"  Optimize for MPS: {OPTIMIZE_FOR_MPS}")
    logger.info(f"  Use mixed precision: {USE_MIXED_PRECISION}")
