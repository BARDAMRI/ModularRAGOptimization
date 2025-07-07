# modules/model_loader.py - GPU OPTIMIZED VERSION
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


def get_optimal_device():
    """
    Determine the best available device for model execution.

    Checks device availability in priority order: FORCE_CPU config -> MPS -> CUDA -> CPU fallback.

    Returns:
        torch.device: The optimal device object (cpu, mps, or cuda)
    """
    if FORCE_CPU:
        logger.info("🔧 CPU forced via config")
        return torch.device("cpu")

    if torch.backends.mps.is_available() and OPTIMIZE_FOR_MPS:
        logger.info("🚀 MPS (Apple Silicon GPU) detected and enabled")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("🚀 CUDA GPU detected")
        return torch.device("cuda")
    else:
        logger.info("💻 Using CPU (no GPU available)")
        return torch.device("cpu")


def load_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Load and optimize a language model with GPU acceleration support.

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

    # Get optimal device
    device = get_optimal_device()

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added padding token to tokenizer.")
    logger.info("Tokenizer loaded successfully.")

    # Try to load as CausalLM first (for text generation)
    try:
        # Device-specific model loading
        if device.type == "mps":
            # MPS-optimized loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,  # MPS requires float32
                low_cpu_mem_usage=True,
                device_map=None  # Don't use device_map with MPS
            )
            logger.info("✅ Loaded model for MPS (Apple Silicon)")

        elif device.type == "cuda":
            # CUDA-optimized loading
            torch_dtype = torch.float16 if USE_MIXED_PRECISION else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            logger.info(f"✅ Loaded model for CUDA with {torch_dtype}")

        else:
            # CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Loaded model for CPU")

        # Move model to device BEFORE optimization
        if device.type != "cuda" or not hasattr(model, 'hf_device_map'):
            # Only move manually if not using device_map
            model = model.to(device)
            logger.info(f"📍 Model moved to {device}")

        logger.info("✅ Loaded AutoModelForCausalLM (supports text generation)")
        return tokenizer, model

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

        # Fallback attempt
        logger.info("🔄 Attempting fallback loading...")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            model = model.to(device)
            logger.info("✅ Fallback loading successful")
            return tokenizer, model
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")
            raise e2


# Replace the get_model_capabilities function in modules/model_loader.py with this:

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
        logger.info("📊 MPS: Memory monitoring limited on Apple Silicon")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(f"📊 CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        logger.info("📊 CPU: No GPU memory to monitor")


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
    logger.info("🧪 Testing GPU-optimized model loading...")

    try:
        tokenizer, model = load_model()
        capabilities = get_model_capabilities(model)

        logger.info("📋 Model loading test results:")
        for key, value in capabilities.items():
            if key == 'can_generate':
                status = "✅" if value else "❌"
            elif key == 'is_causal_lm':
                status = "✅" if value else "⚠️"
            else:
                status = "ℹ️"
            logger.info(f"  {status} {key}: {value}")

        # Monitor memory after loading
        monitor_gpu_memory()

        if capabilities['can_generate']:
            logger.info("🎉 Model supports text generation!")

            # Quick generation test
            device = next(model.parameters()).device
            test_input = tokenizer("Hello", return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **test_input,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Test generation on {device}: '{result}'")

            # Check memory after generation
            monitor_gpu_memory()
        else:
            logger.warning("⚠️ Model does NOT support text generation")

        return capabilities['can_generate']

    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False
