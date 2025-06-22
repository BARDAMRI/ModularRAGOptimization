# config.py

# ✅ מזהה חוקי לשימוש ב-Hugging Face API
MODEL_PATH = "distilbert-base-uncased"

# נתיב לקריאה מקומית של טקסטים
DATA_PATH = "data/public_corpus/"

# סוגי מכשירים
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CPU = "cpu"

# ✅ מודל שפה קל משקל (מחליף את LLAMA)
LLAMA_MODEL_NAME = "distilbert-base-uncased"
LLAMA_MODEL_DIR = "distilbert-base-uncased"  # אם אתה טוען מהאינטרנט, אין צורך בתיקייה מקומית

# ✅ מודל הטמעות מהיר
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# מידע על הקורפוס הציבורי
PUBLIC_CORPUS_DATASET = "wikitext"
PUBLIC_CORPUS_DIR = "data/public_corpus"
LLM_MODEL_NAME = "distilbert-base-uncased"  # מודל שפה ראשי

# פרמטרים לאופטימיזציה
MAX_RETRIES = 3
QUALITY_THRESHOLD = 0.7
RETRIEVER_TOP_K = 2
SIMILARITY_CUTOFF = 0.85
MAX_NEW_TOKENS = 64
NQ_SAMPLE_SIZE = 5

# ✅ מקור ברירת מחדל לנתונים מהאינטרנט
DEFAULT_HF_DATASET = "wikipedia"
DEFAULT_HF_CONFIG = "20220301.en"
INDEX_SOURCE_URL = "wikipedia:20220301.en"