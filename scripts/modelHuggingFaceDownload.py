# modelHuggingFaceDownload.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from configurations.config import MODEL_PATH
from configurations.config import LLAMA_MODEL_DIR

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")

# Save them inside your project
model.save_pretrained(LLAMA_MODEL_DIR)
tokenizer.save_pretrained(LLAMA_MODEL_DIR)

print("> Model downloaded successfully!")
