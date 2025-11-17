# config.py - Light QA-Ready Configuration for Low-RAM/No-GPU systems
from utility.distance_metrics import DistanceMetric, StoringMethod

# ==========================
# ✅ ACTIVE MODEL CONFIGURATION (for QA on CPU / M1 / 16GB RAM)
# ==========================

MODEL_PATH = "tiiuae/falcon-rw-1b"  # 🐦 Falcon 1B - very lightweight, extremely fast, basic QA
EVALUATION_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # MiniLM-6B for evaluation - fast and efficient
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
HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" ✅ Default: Fast and efficient, good for low-resource environments.
# HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # ⭐ Recommended next step. Good balance of accuracy and speed. Significantly better than MiniLM for semantic understanding.
# HF_MODEL_NAME = "BAAI/bge-small-en-v1.5" # ✨ High accuracy for its size. Part of the BGE family, known for strong retrieval performance.
# HF_MODEL_NAME = "BAAI/bge-base-en-v1.5" # 🔥 Even higher accuracy than bge-small, but slower and requires more RAM.
# HF_MODEL_NAME = "BAAI/bge-large-en-v1.5" # 🚀 State-of-the-art accuracy, but very large (~1.3GB) and slow on CPU/MPS. Only for high-end GPUs.

# ==========================
# QA DATASET CONFIGURATION
# ==========================

QA_DATASETS = {
    "squad": {
        "config": "plain_text",
        "description": "📘 SQuAD - open QA dataset",
        "corpus_source": "wikimedia/wikipedia",
        "corpus_config": "20220301.en"
    },
    "cais/mmlu": {
        "config": "all",
        "description": "🌐 MMLU - multitask multiple-choice (57 domains)",
        "corpus_source": "wikimedia/wikipedia",
        "corpus_config": "20220301.en"
    },
    "openbookqa": {
        "config": "main",
        "description": "📖 OpenBookQA - Science + commonsense multiple choice",
        "corpus_source": "allenai/openbookqa"
    },
    "commonsense_qa": {
        "config": None,
        "description": "💡 CommonsenseQA - Commonsense reasoning",
        "corpus_source": "conceptnet/conceptnet5"
    },
    "bigbio/med_qa": {
        "config": "med_qa_en",
        "description": "🏥 MedQA-US - USMLE medical exam questions",
        "corpus_source": "uiyunkim-hub/pubmed-abstract"
    },
    "medmcqa": {
        "config": "train",
        "description": "🩺 MedMCQA - Indian medical exam questions",
        "corpus_source": "uiyunkim-hub/pubmed-abstract"
    },
    "qiaojin/PubMedQA": {
        "config": "pqa_labeled",
        "description": "📄 PubMedQA - Literature-based biomedical QA",
        "corpus_source": "pubmed_selected_articles"
    },
    "bioasq": {
        "config": "task2b",
        "description": "🧬 BioASQ - Biomedical yes/no QA",
        "corpus_source": "uiyunkim-hub/pubmed-abstract"
    }
}

# 🔑 Active dataset key
ACTIVE_QA_DATASET = "qiaojin/PubMedQA"  # 📄 PubMedQA - Literature-based biomedical QA

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
RETRIEVER_TOP_K = 10
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 50
NQ_SAMPLE_SIZE = -1  # Use -1 for full dataset
TEMPERATURE = 0.05

# === DATA SOURCE ===
# The source URL for indexing. Can point to Hugging Face datasets or web URLs.
# Automatically select corpus based on active dataset
INDEX_SOURCE_URL = QA_DATASETS.get(ACTIVE_QA_DATASET, {}).get("corpus_source", "wikimedia/wikipedia")

# Optional alternative data sources (uncomment to switch)
# INDEX_SOURCE_URL = "wikimedia/wikipedia" # Default Wikipedia dump.
# ✅ Advantage: Broad general knowledge, widely used for QA datasets. Use for general-purpose testing or when a diverse knowledge base is needed.
# INDEX_SOURCE_URL = "hf:huggingface/documentation" # Example: Hugging Face documentation dataset.
# ✅ Advantage: High-quality, structured, domain-specific text. Use when your queries are about NLP, ML, or Hugging Face ecosystem.
# INDEX_SOURCE_URL = "hf:ag_news" # Example: AG News classification dataset.
# ✅ Advantage: Well-categorized, good for text classification/short news articles. Use when dealing with news topics or short, distinct documents.
# INDEX_SOURCE_URL = "https://www.gutenberg.org/files/1342/1342-0.txt" # Example: A direct URL to a text file (e.g., Pride and Prejudice).
# ✅ Advantage: Simple, direct access to specific public domain texts. Use for testing with a single, known document or literary analysis.
# INDEX_SOURCE_URL = "data/my_local_documents" # Example: A local directory containing your own text files (ensure it exists).
# ✅ Advantage: Full control over content, private data. Use for domain-specific or confidential information.
# === GPU OPTIMIZATION SETTINGS ===
FORCE_CPU = False  # Set to True to force CPU usage
OPTIMIZE_FOR_MPS = True  # Set to True to optimize for Apple Silicon (MPS) device
MAX_GPU_MEMORY_GB = 8
USE_MIXED_PRECISION = False
# Set to True if you have a compatible NVIDIA GPU (e.g., RTX series)
# and experiencing out-of-memory issues or want faster training/inference.
# Keep False for CPU/MPS as it generally offers no benefit or can cause issues if not fully supported.
# For most consumer GPUs, float16 (half-precision) is typically used when True.

# === TRILATERATION RETRIEVER SETTINGS ===
TRILATERATION_ITERATIVE = True  # Whether to use iterative refinement (add selected docs to anchors)
TRILATERATION_MAX_REFINES = 3  # Maximum number of refinement iterations
TRILATERATION_CONVERGENCE_TOL = 1e-4  # Convergence tolerance for x* movement between iterations

DEFAULT_STORING_METHOD = StoringMethod.CHROMA
DEFAULT_DISTANCE_METRIC = DistanceMetric.COSINE
