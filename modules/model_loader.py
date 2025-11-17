# modules/model_loader.py - Complete version with performance monitoring
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from sentence_transformers import CrossEncoder
from configurations.config import EVALUATION_MODEL_NAME
from configurations.config import MODEL_PATH, FORCE_CPU, OPTIMIZE_FOR_MPS, USE_MIXED_PRECISION
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


def load_evaluator_model() -> CrossEncoder:
    """
    Load a lightweight evaluator model (e.g., cross-encoder) for scoring query-document pairs.
    """
    logger.info(f"Loading evaluator model: {EVALUATION_MODEL_NAME} ...")
    try:
        model = CrossEncoder(EVALUATION_MODEL_NAME)
        logger.info("Evaluator model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load evaluator model '{EVALUATION_MODEL_NAME}': {e}")
        raise e


def generate_text_mps_safe(model, inputs, device, tokenizer, max_tokens: int = 64, temperature: float = 0.07):
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
                    temperature=temperature,
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
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
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
                temperature=temperature,
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
        embedding_model=None
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
