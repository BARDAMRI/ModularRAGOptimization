from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_PATH

MODEL_PATH = "/Users/bardamri/PycharmProjects/ModularRAGOptimization/models/Llama3.2-8B"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto"  # Auto-detect CPU or GPU
    )
    return tokenizer, model
