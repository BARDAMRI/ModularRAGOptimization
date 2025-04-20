# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_PATH
import torch


def load_model():
    print(f"\n> Loading Model {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = torch.compile(model)  # Optimizes execution speed
    print(f"\n> The Model was loaded successfully")
    return tokenizer, model

