import os

from chromadb import PersistentClient

from utility.vector_db_utils import parse_source_path

# === הגדרות בסיס ===
base = os.path.dirname(os.getcwd())
DB_DIR = os.path.join(base,  "storage/chroma/pubmed_abstracts/cosine")
source_type, parsed_name = parse_source_path('/Users/bardamri/PycharmProjects/ModularRAGOptimization/data/pubmed_abstracts')
collection_name = f"collection_{parsed_name.replace(':', '_')}"
EXPECTED_DOCS = 3_214_773
DIMS = 768  # ממד האמבדינג
BYTES_PER_FLOAT = 4  # float32


def get_collection_count():
    """Count how many embeddings/documents already exist."""
    try:
        client = PersistentClient(path=DB_DIR)
        col = client.get_collection(collection_name)
        return col.count()
    except Exception as e:
        print(f"⚠️ Failed to count collection items: {e}")
        return None


def get_sqlite_size():
    """Return current size of the sqlite3 file in GB."""
    sqlite_path = os.path.join(DB_DIR, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        return os.path.getsize(sqlite_path) / (1024 ** 3)
    return 0.0


def estimate_expected_size(avg_doc_bytes=1000, overhead_factor=1.25):
    """
    Estimate final size (GB):
    - avg_doc_bytes: גודל ממוצע למחרוזת טקסט
    - overhead_factor: תקורה של מטא-דאטה ואינדקסים (1.2–1.3 לרוב)
    """
    embeddings_bytes = EXPECTED_DOCS * DIMS * BYTES_PER_FLOAT
    text_bytes = EXPECTED_DOCS * avg_doc_bytes
    total_bytes = (embeddings_bytes + text_bytes) * overhead_factor
    return total_bytes / (1024 ** 3)


def main():
    count = get_collection_count()
    sqlite_size = get_sqlite_size()
    expected_total_gb = estimate_expected_size(avg_doc_bytes=900, overhead_factor=1.25)

    print("===================================================")
    print("📊 Vector DB Build Progress Estimator")
    print("===================================================")
    print(f"Total expected docs:   {EXPECTED_DOCS:,}")
    print(f"Current DB size:       {sqlite_size:.2f} GB")
    if count:
        print(f"Indexed documents:     {count:,}")
        progress_docs = count / EXPECTED_DOCS
    else:
        progress_docs = 0
        print("Indexed documents:     (unknown)")

    progress_size = min(sqlite_size / expected_total_gb, 1.0)
    avg_progress = (progress_docs + progress_size) / 2

    print(f"Estimated total size:  {expected_total_gb:.2f} GB")
    print(f"Progress by doc count: {progress_docs * 100:.1f}%")
    print(f"Progress by file size: {progress_size * 100:.1f}%")
    print("---------------------------------------------------")
    print(f"✅ Estimated completion: {avg_progress * 100:.1f}%")
    print(f"🕒 Remaining fraction:   {(1 - avg_progress) * 100:.1f}%")
    print("===================================================")


if __name__ == "__main__":
    main()
