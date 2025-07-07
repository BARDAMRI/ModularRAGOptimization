# modelHuggingFaceDownload.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from configurations.config import LLAMA_MODEL_NAME
from configurations.config import LLAMA_MODEL_DIR
# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, torch_dtype="auto")

# Save them inside your project
model.save_pretrained(LLAMA_MODEL_DIR)
tokenizer.save_pretrained(LLAMA_MODEL_DIR)

print("> Model downloaded successfully!")