# config.py - Light QA-Ready Configuration for Low-RAM/No-GPU systems

# ==========================
# ✅ ACTIVE MODEL CONFIGURATION (for QA on CPU / M1 / 16GB RAM)
# ==========================

MODEL_PATH = "tiiuae/falcon-rw-1b"                     # 🐦 Falcon 1B - very lightweight, extremely fast, basic QA

# ==========================
# Optional lightweight models (uncomment to switch)
# ==========================
# MODEL_PATH = "microsoft/phi-2"  # 🧠 Phi-2 (2.7B) - small, high-quality, works well on CPU with low RAM
# MODEL_PATH = "EleutherAI/gpt-neo-1.3B"                 # 🤖 GPT-Neo 1.3B - simple, good compatibility, fair QA
# MODEL_PATH = "openchat/openchat-3.5-0106"              # 💬 OpenChat 3.5 (3.5B) - solid QA/dialogue, quantize for better speed

# ==========================
# Do NOT use these unless you have 24GB+ VRAM or offloading infra
# ==========================
# MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"       # ⚠️ Heavy - requires 14GB+ RAM/VRAM
# MODEL_PATH = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  # ⚠️ Heavy but high quality (7B)
# MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"            # ⚠️ 7B - accurate, not suitable for low RAM
# MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"     # ❌ 13B+, MoE - too large
# MODEL_PATH = "meta-llama/Llama-2-13b-chat-hf"           # ❌ 13B - very heavy

# ==========================
# Embedding model
# ==========================
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === DATA PATHS ===
DATA_PATH = "data/public_corpus/"
PUBLIC_CORPUS_DATASET = "wikitext"
PUBLIC_CORPUS_DIR = "data/public_corpus"

# === DEVICE CONFIGURATION ===
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# === LLM SETTINGS ===
LLM_MODEL_NAME = MODEL_PATH
LLAMA_MODEL_DIR = MODEL_PATH

# === OPTIMIZATION PARAMETERS ===
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 50
NQ_SAMPLE_SIZE = 5
TEMPERATURE = 0.05
# === DATA SOURCE ===
DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"

# === GPU OPTIMIZATION SETTINGS ===
FORCE_CPU = False  # Set to True to force CPU usage
OPTIMIZE_FOR_MPS = True
MAX_GPU_MEMORY_GB = 8
USE_MIXED_PRECISION = False  # Recommended: False for MPS