from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
import torch
from config import MODEL_PATH
from typing import Tuple
from utility.logger import logger


def load_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    logger.info(f"Loading Model {MODEL_PATH}...")

    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info("Tokenizer loaded successfully.")

    architectures = getattr(config, "architectures", [])

    if any("CausalLM" in arch for arch in architectures):
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        logger.info("Loaded AutoModelForCausalLM.")
    elif any("SequenceClassification" in arch for arch in architectures):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        logger.info("Loaded AutoModelForSequenceClassification.")
    else:
        model = AutoModel.from_pretrained(MODEL_PATH)
        logger.info("Loaded generic AutoModel.")

    if torch.__version__.startswith("2"):
        try:
            model = torch.compile(model)
            logger.info("Model optimized using torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    return tokenizer, model
