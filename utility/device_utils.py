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
