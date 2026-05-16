# config.py - Light QA-Ready Configuration for Low-RAM/No-GPU systems
import os
from pathlib import Path

from dotenv import load_dotenv

from utility.distance_metrics import DistanceMetric, StoringMethod

# Load project-root .env before reading os.environ-backed settings below.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ==========================
# ✅ ACTIVE MODEL CONFIGURATION (for QA on CPU / M1 / 16GB RAM)
# ==========================

# MODEL_PATH = "tiiuae/falcon-rw-1b"  # 🐦 Falcon 1B - very lightweight, extremely fast, basic QA
EVALUATION_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # MiniLM-6B for evaluation - fast and efficient
EVALUATOR_TYPE = "Both"  # Options: "LLM" or "CrossEncoder" or "Both"

# ==========================
# Optional lightweight models (uncomment to switch)
# ==========================
# MODEL_PATH = "microsoft/phi-2"  # 🧠 Phi-2 (2.7B) - small, high-quality, works well on CPU with low RAM
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_PATH = "EleutherAI/gpt-neo-1.3B"                 # 🤖 GPT-Neo 1.3B - simple, good compatibility, fair QA
# MODEL_PATH = "openchat/openchat-3.5-0106"              # 💬 OpenChat 3.5 (3.5B) - solid QA/dialogue, quantize for better speed


GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyAmXU4-f853LIVpa4S66aAO6yIp9XAUCEs",
)  # Google Gemini API key (override via .env)
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
RUN_RANDOM_SCENARIO = True  # If True, runs random scenarios for testing
# === DEVICE CONFIGURATION ===
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# === LLM SETTINGS ===
LLM_MODEL_NAME = MODEL_PATH
LLAMA_MODEL_DIR = MODEL_PATH

# ==========================
# Global Correlation LLM Provider
# ==========================
# Options:
# - "gemini"     -> Gemini Batch API (offline JSONL generation + harvester)
# - "ollama"     -> BGU cis-ollama (live scoring; no Gemini cost)
# - "nvidia_ih"  -> NVIDIA Inference Hub (HTTPS generateContent; set IH_API_KEY)
CORRELATION_LLM_PROVIDER = "nvidia_ih"

# Gemini Batch model id (keep in sync with Batch API calls + LLM gateway batch limits).
# ``gemini-2.0-flash`` is deprecated; default to 2.5 Flash-Lite (high RPM, unlimited RPD on typical tiers).
CORRELATION_GEMINI_BATCH_MODEL = "gemini-2.5-flash-lite"

# --- Gemini API rate limits (official reference) --------------------------------
# Primary doc: https://ai.google.dev/gemini-api/docs/rate-limits
# - Interactive traffic (e.g. generateContent): RPM, TPM (input), RPD (and model-specific caps like IPM).
#   Limits are per *project*; RPD resets at **midnight Pacific**. Preview/experimental models are tighter.
# - **Batch API** quotas are *separate*: concurrent batch requests, per-job input file size, Files pool,
#   and per-model **enqueued input tokens** across all active batch jobs (Tier 1 vs Tier 2–3 columns in the doc).
# - **Priority inference** uses ~0.3× the standard interactive limits for the same model/tier (same doc page).
# - Concrete ceilings also appear under Batch mode in the experiment module
#   (``GEMINI_BATCH_*`` in ``experiments/global_correlation_experiment.py``): 100 concurrent jobs, 2 GiB input, 20 GiB Files.
#
# The tables below copy Google's **max enqueued batch input tokens** (not enforced by ``LLMGateway`` — for planning only).
# Keys must match the **exact** ``model`` string you pass to Batch. Tier unknown ⇒ check AI Studio; set tier for helpers.
GEMINI_API_BILLING_TIER = int(os.environ.get("GEMINI_API_BILLING_TIER", "2"))
# Use ``1`` for the Tier 1 batch enqueued-token table; ``2`` or ``3`` for the Tier 2–3 table.

GEMINI_BATCH_ENQUEUED_INPUT_TOKENS_TIER_1: dict[str, int] = {
    "gemini-3.1-pro-preview": 5_000_000,
    "gemini-3.1-flash-lite": 10_000_000,
    "gemini-3.1-flash-lite-preview": 10_000_000,
    "gemini-3-flash-preview": 3_000_000,
    "gemini-2.5-pro": 5_000_000,
    "gemini-2.5-pro-preview-tts": 25_000,
    "gemini-2.5-flash": 3_000_000,
    "gemini-2.5-flash-preview": 3_000_000,
    "gemini-2.5-flash-image-preview": 3_000_000,
    "gemini-2.5-flash-preview-tts": 100_000,
    "gemini-2.5-flash-lite": 10_000_000,
    "gemini-2.5-flash-lite-preview": 10_000_000,
    "gemini-2.0-flash": 10_000_000,
    "gemini-2.0-flash-image": 3_000_000,
    "gemini-2.0-flash-lite": 10_000_000,
    "gemini-3.1-flash-image-preview": 1_000_000,
    "gemini-3-pro-image-preview": 2_000_000,
    "gemini-embedding-001": 500_000,
}

GEMINI_BATCH_ENQUEUED_INPUT_TOKENS_TIER_2_3: dict[str, int] = {
    "gemini-3.1-pro-preview": 500_000_000,
    "gemini-3.1-flash-lite": 500_000_000,
    "gemini-3.1-flash-lite-preview": 500_000_000,
    "gemini-3.1-flash-preview": 400_000_000,
    "gemini-2.5-pro": 500_000_000,
    "gemini-2.5-pro-preview-tts": 100_000,
    "gemini-2.5-flash": 400_000_000,
    "gemini-2.5-flash-preview": 400_000_000,
    "gemini-2.5-flash-image-preview": 400_000_000,
    "gemini-2.5-flash-preview-tts": 100_000,
    "gemini-2.5-flash-lite": 500_000_000,
    "gemini-2.5-flash-lite-preview": 500_000_000,
    "gemini-2.0-flash": 1_000_000_000,
    "gemini-2.0-flash-image": 400_000_000,
    "gemini-2.0-flash-lite": 1_000_000_000,
    "gemini-3.1-flash-image-preview": 250_000_000,
    "gemini-3-pro-image-preview": 270_000_000,
    "gemini-embedding-001": 5_000_000,
}


def gemini_batch_enqueued_input_token_cap(model: str) -> int | None:
    """Return the doc max *enqueued batch input tokens* for ``model``, or ``None`` if unknown."""
    m = str(model).strip()
    if GEMINI_API_BILLING_TIER <= 1:
        return GEMINI_BATCH_ENQUEUED_INPUT_TOKENS_TIER_1.get(m)
    return GEMINI_BATCH_ENQUEUED_INPUT_TOKENS_TIER_2_3.get(m)

# ==========================
# LLM Gateway — per-provider rate limits
# ==========================
# ``None`` / missing ``rpm`` / missing provider or model ⇒ **no throttle** for that rule.
#
# Key resolution order (first match wins):
#   1. exact model name             e.g. "gemini-2.5-flash-lite"
#   2. "<kind>:<model>"             e.g. "regular:gemini-2.5-flash-lite"
#   3. "<kind>:*"                   e.g. "batch:*"
#   4. "__default__"
#
# Supported limit fields (pick one per entry):
#   "rpm"              requests per minute
#   "rps"              requests per second
#   "min_interval_s"   exact minimum seconds between requests
LLM_NO_RATE_LIMIT = None  # explicit documented sentinel

LLM_GATEWAY_RATE_LIMITS: dict = {
    # ------------------------------------------------------------------
    # Ollama — BGU CIS cluster (https://cis-ollama.auth.ad.bgu.ac.il)
    # No server-enforced rate limit; tune to observed cluster headroom.
    # Concurrency is governed by OLLAMA_MAX_CONCURRENT_REQUESTS (async
    # semaphore in the experiment), not by the gateway RPM below.
    # Uncomment and lower if you see connection timeouts under load.
    # ------------------------------------------------------------------
    "ollama": {
        # "__default__": {"rpm": 120},  # ~2 req/s; conservative for shared cluster
    },

    # ------------------------------------------------------------------
    # Google Gemini API — text-out peaks (AI Studio / API “Rate limits”, 2026-05 snapshot).
    # Source: aistudio.google.com → API → Rate limits (per model; your tier may differ).
    #
    # Product name (dashboard)   Model API ID (examples)          RPM    TPM      RPD
    # Gemini 2 Flash (dep.)      gemini-2.0-flash              10 000  10 M     Unlimited
    # Gemini 2 Flash Lite (dep.) gemini-2.0-flash-lite         20 000  10 M     Unlimited
    # Gemini 2.5 Flash           gemini-2.5-flash               2 000   3 M     100 K
    # Gemini 2.5 Flash Lite      gemini-2.5-flash-lite        10 000  10 M     Unlimited
    # Gemini 2.5 Pro             gemini-2.5-pro                1 000   5 M      50 K
    # Gemini 3 Flash             gemini-3-flash-preview        2 000   3 M     100 K
    # Gemini 3.1 Flash Lite      gemini-3.1-flash-lite        10 000  10 M     350 K
    # Gemini 3.1 Pro Preview     gemini-3.1-pro-preview        1 000   5 M      50 K
    #
    # The gateway only paces RPM / RPS / min_interval_s. TPM and RPD are not enforced here;
    # watch the dashboard if you run very large batch jobs (especially 350K RPD caps).
    # Batch job *submission* pacing is ``batch:*`` above; **enqueued batch input-token**
    # ceilings per model/tier are documented in ``GEMINI_BATCH_ENQUEUED_INPUT_TOKENS_*`` (not auto-enforced).
    #
    # Key format "regular:<model>" applies only to live generate_content
    # calls; "batch:*" catches all Batch API job submissions.
    # ------------------------------------------------------------------
    "gemini": {
        # ── Live (regular) generate_content ────────────────────────────
        "regular:gemini-2.0-flash":      {"rpm": 10_000},
        "regular:gemini-2.0-flash-lite": {"rpm": 20_000},
        "regular:gemini-2.5-flash":      {"rpm": 2_000},
        "regular:gemini-2.5-flash-lite": {"rpm": 10_000},
        "regular:gemini-2.5-pro":        {"rpm": 1_000},
        "regular:gemini-3-flash-preview": {"rpm": 2_000},
        "regular:gemini-3.1-flash-lite": {"rpm": 10_000},
        "regular:gemini-3.1-flash-lite-preview": {"rpm": 10_000},
        "regular:gemini-3.1-pro-preview": {"rpm": 1_000},
        "regular:gemini-3.1-pro-preview-customtools": {"rpm": 1_000},
        # ── Batch job submissions ────────────────────────────────────────
        # Each submission is one POST per JSONL file, not one per row.
        # 60 rpm is generous for job-level submits; raises on demand.
        "batch:*": {"rpm": 60},
        # ── Fallback for any unlisted model ─────────────────────────────
        "__default__": {"rpm": LLM_NO_RATE_LIMIT},
    },

    # ------------------------------------------------------------------
    # NVIDIA Inference Hub  (inference-api.nvidia.com)
    # Free tier: 40 RPM.  Higher tiers available on request (up to 200+).
    # Source: NVIDIA Developer forums + NIM documentation (2025).
    # The default covers all models routed through NVIDIA_IH_MODEL.
    # ------------------------------------------------------------------
    "nvidia_ih": {
        # Model key matches the full NVIDIA_IH_MODEL path, e.g.:
        #   "gcp/google/gemini-2.5-flash-lite": {"rpm": 40},
        "__default__": {"rpm": 40},  # free tier; raise after quota upgrade
    },
}

# --- NVIDIA Inference Hub (Vertex-style generateContent) ---
# Override URL entirely with NVIDIA_IH_URL_TEMPLATE env if your project path differs.
NVIDIA_IH_MODEL = os.environ.get("IH_MODEL", "gcp/google/gemini-2.5-flash-lite")
NVIDIA_IH_GENERATE_URL_TEMPLATE = os.environ.get(
    "NVIDIA_IH_URL_TEMPLATE",
    "https://inference-api.nvidia.com/vertex_ai/v1/projects/"
    "nv-gcpllmgwit-20250411173346/locations/global/publishers/google/models/"
    "{model}:generateContent",
)
NVIDIA_IH_TIMEOUT_S = float(os.environ.get("NVIDIA_IH_TIMEOUT_S", "180"))
NVIDIA_IH_API_KEY = (
    os.environ.get("IH_API_KEY")
    or os.environ.get("NVIDIA_IH_API_KEY")
    or os.environ.get("API_KEY")
    or ""
)

# BGU CIS Ollama endpoint (see AI_ollama_Chat_ guide + your examples)
OLLAMA_HOST = "https://cis-ollama.auth.ad.bgu.ac.il"

# Example model names from your guide: "llama3.2", "gpt-oss:20b"
# Balanced quality/speed for structured scoring responses.
CORRELATION_OLLAMA_MODEL = "Qwen3.5:4B"

# cis-ollama uses internal TLS; disable verification by default (matches examples).
OLLAMA_VERIFY_SSL = False

# Safety: request timeout (seconds) for Ollama calls.
OLLAMA_TIMEOUT_S = 180

# Fail-fast: stop experiment quickly on cluster connectivity failures.
OLLAMA_FAIL_FAST_ON_CONNECTION_ERROR = True

# Number of concurrent Ollama requests per query in live scoring mode.
# Higher values improve throughput but can increase timeout risk under cluster load.
OLLAMA_MAX_CONCURRENT_REQUESTS = 8

# Number of documents scored per Ollama request in global correlation experiment.
OLLAMA_DOCS_PER_REQUEST = 10


def correlation_live_model_name() -> str:
    """Model id for live scoring on Ollama or NVIDIA Inference Hub (not Gemini Batch)."""
    p = str(CORRELATION_LLM_PROVIDER).strip().lower()
    if p in ("nvidia_ih", "nvidia", "inference_hub"):
        return str(NVIDIA_IH_MODEL)
    return CORRELATION_OLLAMA_MODEL


def configured_correlation_provider() -> str:
    """Current provider string (always reads live config, including runtime overrides)."""
    return str(CORRELATION_LLM_PROVIDER).strip().lower()


# ==========================
# Global Correlation small-batch limits (Pilot)
# ==========================
# Limits docs per query in Pilot mode to keep validation runs fast and observable.
CORRELATION_PILOT_MAX_DOCS_PER_QUERY = 40

# ==========================
# Staged / additive-pool run parameters
# ==========================
# Documents added per stream (ranked + random) per query per stage.
STAGING_STRIDE = 5
# Max ranked neighbours retrieved per GT embedding across all stages.
STAGING_MAX_RANKED_PER_GT = 100
# Max ranked neighbours retrieved per query embedding across all stages.
STAGING_MAX_RANKED_PER_QUERY = 100
# Max random corpus samples per query across all stages.
STAGING_MAX_RANDOM_PER_QUERY = 100

# === OPTIMIZATION PARAMETERS ===
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 1000
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 50
NQ_SAMPLE_SIZE = -1  # Use -1 for full dataset
TEMPERATURE = 0.05

# ======== TRILATERATION SETTINGS ========
# Hyperparameters for the metric least-squares trilateration variant
LIN_SOLVER_MAX_ITERATIONS = 20  # Maximum gradient-descent iterations in metric least-squares solver
LIN_SOLVER_STEP_SIZE = 0.1  # Learning rate used in optimization updates
LIN_SOLVER_TOLERANCE = 1e-4  # Stop condition: minimal gradient magnitude change
LIN_SOLVER_EPSILON = 1e-8  # Numerical stability constant preventing division by zero
DISTANCE_SCALE_GAMMA = 1.0  # Scale mapping similarity → pseudo-radius (normalizes distances)
NUM_BASE_ANCHORS = 4  # Number of initial anchors for trilateration
TRILATERATION_MODE = "metric_least_squares"  # Options: "weighted_centroid" or "metric_least_squares"

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
