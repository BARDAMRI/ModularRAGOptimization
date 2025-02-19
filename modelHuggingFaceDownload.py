from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

# Save them inside your project
model.save_pretrained("models/Llama3.2-8B")
tokenizer.save_pretrained("models/Llama3.2-8B")

print("✅ Model downloaded successfully!")