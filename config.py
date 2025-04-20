# config.py
MODEL_PATH = "models/Llama3.2-8B"
DATA_PATH = "data/public_corpus/"

DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
LLAMA_MODEL_DIR = "models/Llama3.2-8B"
HF_MODEL_NAME = "all-MiniLM-L6-v2"
PUBLIC_CORPUS_DATASET = "wikitext-2-raw-v1"
PUBLIC_CORPUS_DIR = "data/public_corpus"

MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 64  # For a fuller sentence or paragraph should be set here to 128 short factual answers

DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"
