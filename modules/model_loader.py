# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from config import MODEL_PATH
from utility.logger import logger  # Import logger
import torch
from typing import Tuple


def load_model() -> Tuple[GPT2TokenizerFast, AutoModelForCausalLM]:
    """
    Load the model and tokenizer from the specified path.
    This function initializes the tokenizer and model, optimizes the model using torch.compile,
    and returns both the tokenizer and model.
    It uses GPT2TokenizerFast for efficient tokenization.

    Raises:

    Returns
        Tuple[GPT2TokenizerFast, AutoModelForCausalLM]: A tuple containing the tokenizer and model.

    """
    logger.info(f"Loading Model {MODEL_PATH}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)  # Updated to GPT2TokenizerFast
    logger.info("Tokenizer loaded successfully.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info("Model loaded successfully.")

    model = torch.compile(model)  # Optimizes execution speed
    logger.info("Model optimized using torch.compile.")

    logger.info("Model loading process completed successfully.")
    return tokenizer, model
