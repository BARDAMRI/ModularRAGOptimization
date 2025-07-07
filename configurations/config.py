# config.py - FIXED VERSION
# ✅ This replaces your current config.py

# === MODEL CONFIGURATION ===
# ✅ Use a proper text generation model instead of DistilBERT
MODEL_PATH = "distilgpt2"
# Alternative options (uncomment one if you prefer):
# MODEL_PATH = "microsoft/DialoGPT-small"  # Good for conversations, lightweight. 30% more complex than distilgpt2.
# MODEL_PATH = "gpt2"  # Classic, reliable
# MODEL_PATH = "distilgpt2"  # Faster, smaller version

# ✅ Keep your embedding model (this one is correct)
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === DATA PATHS ===
DATA_PATH = "data/public_corpus/"
PUBLIC_CORPUS_DATASET = "wikitext"
PUBLIC_CORPUS_DIR = "data/public_corpus"

# === DEVICE CONFIGURATION ===
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# === LLM SETTINGS ===
LLM_MODEL_NAME = MODEL_PATH  # Use the same model for consistency
LLAMA_MODEL_DIR = MODEL_PATH  # Updated to match

# === OPTIMIZATION PARAMETERS ===
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 64
NQ_SAMPLE_SIZE = 5

# === DATA SOURCE ===
DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"

# === GPU OPTIMIZATION SETTINGS ===
FORCE_CPU = False  # Set to True to force CPU usage
OPTIMIZE_FOR_MPS = True  # Apple Silicon optimizations
MAX_GPU_MEMORY_GB = 8  # Adjust based on your hardware
USE_MIXED_PRECISION = False  # Enable for CUDA, disable for MPS
