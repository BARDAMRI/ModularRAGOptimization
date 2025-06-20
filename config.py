# config.py
MODEL_PATH = "models/DistilBERT"
DATA_PATH = "data/public_corpus/"

DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# Lightweight models
LLAMA_MODEL_NAME = "distilbert-base-uncased"  # Replacing heavy Llama model
LLAMA_MODEL_DIR = "models/DistilBERT"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
PUBLIC_CORPUS_DATASET = "wikitext-2-raw-v1"
PUBLIC_CORPUS_DIR = "data/public_corpus"
LLM_MODEL_NAME = "distilbert-base-uncased"  # Lightweight LLM

MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 64  # For short factual answers
NQ_SAMPLE_SIZE = 5

DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"