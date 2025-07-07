# === File: combine.py ===
import os

root_dir = "/Users/bardamri/PycharmProjects/ModularRAGOptimization"
output_file_path = "combined_project.py"

# רשימת תיקיות שאנחנו רוצים להתעלם מהן
excluded_dirs = {"__pycache__", ".venv", "env", ".git", ".idea", "build", "dist", "tests", "user_query_datasets"}

python_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    # סינון תיקיות לא רלוונטיות (in-place)
    dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

    for filename in filenames:
        if filename.endswith(".py") and filename != os.path.basename(output_file_path):
            full_path = os.path.join(dirpath, filename)
            python_files.append(full_path)

# כתיבת הקבצים המאושרים לקובץ אחד
with open(output_file_path, "w", encoding="utf-8") as outfile:
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, root_dir)
        outfile.write(f"# === File: {rel_path} ===\n")
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
        except UnicodeDecodeError:
            print(f"⚠️  קובץ לא נקרא ב-UTF-8: {file_path}, עובר ל-latin-1")
            with open(file_path, "r", encoding="latin-1") as infile:
                content = infile.read()
        outfile.write(content)
        outfile.write("\n\n")

print(f"✅ נוצר קובץ מאוחד רק עם הקוד שלך: {output_file_path}")

# === File: docsDownloader.py ===
import subprocess
import os


def download_llamaindex_docs(url, destination_folder):
    """
    Downloads the LlamaIndex documentation locally for offline use using httrack instead of wget.
    """
    os.makedirs(destination_folder, exist_ok=True)
    command = [
        "httrack",
        url,
        "-O", destination_folder,
        "--mirror",
        "--keep-alive",
        "--quiet"
    ]

    print(f"Downloading LlamaIndex documentation into '{destination_folder}'...")
    subprocess.run(command)
    print(f"Download complete! You can browse the docs offline from '{destination_folder}'.")


URL = input('insert URL: \t')
name = input('insert Name: \t')
if name and len(name) > 0 and URL and len(URL) > 0:
    download_llamaindex_docs(url=URL, destination_folder=name)


# === File: code_testing.py ===
# ============== Getting the type of the vector_db. ============

from llama_index.core.indices.base import BaseIndex
from configurations.config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from llama_index.core import VectorStoreIndex

# ============ Getting the type of the vector_db. ============
vector_db, embedding_space = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
print(type(vector_db))  # Prints the type of vector_db

# Example: Using isinstance() to check if it's a specific type
if isinstance(vector_db, VectorStoreIndex):
    print("vector_db is a VectorStoreIndex")
    print(type(vector_db.vector_store))  # Prints the type of the vector store
elif isinstance(vector_db, BaseIndex):
    print("vector_db is a BaseIndex")
else:
    print(f"Unknown type: {type(vector_db)}")

# Example: Using dir() to inspect the object
print(dir(vector_db))  # Lists all attributes and methods of vector_db

# ============ Getting the type of the tokenizer.============


# from transformers import AutoTokenizer, GPT2TokenizerFast
#
# # Example: Initialize tokenizer
# MODEL_PATH = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#
# # Check the type of the tokenizer
# print(type(tokenizer))  # Prints the type of tokenizer
#
# # Verify if the tokenizer is an instance of GPT2TokenizerFast
# if isinstance(tokenizer, GPT2TokenizerFast):
#     print("tokenizer is a GPT2TokenizerFast")
# elif isinstance(tokenizer, AutoTokenizer):
#     print("tokenizer is an AutoTokenizer")
# else:
#     print("Unknown tokenizer type")


# ============ Getting the type of the vector_db.============
# vector_db = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
# print(type(vector_db))  # Prints the type of vector_db
#
# # Example: Using isinstance() to check if it's a specific type
#
#
# if isinstance(vector_db, VectorStoreIndex):
#     print("vector_db is a VectorStoreIndex")
# elif isinstance(vector_db, BaseIndex):
#     print("vector_db is a BaseIndex")
# else:
#     print("Unknown type")
#
# # Example: Using dir() to inspect the object
# print(dir(vector_db))  # Lists all attributes and methods of vector_db


# ============ Generating a response with refined parameters.============
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# # Initialization
# MODEL_PATH = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prompt = "Hello, how are you?"
#
# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
# # Generate output with refined parameters
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=30,  # Limit response length
#     do_sample=True,
#     temperature=0.6,    # Reduce randomness
#     top_k=20,           # Stricter vocabulary sampling
#     top_p=0.8,          # Nucleus sampling
#     repetition_penalty=2.0,  # Penalize repetition more aggressively
#     pad_token_id=tokenizer.eos_token_id
# )
#
# # Decode and print response
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(response[0])


# === File: main.py ===
# main.py
from modules.model_loader import load_model
from modules.query import query_model
from modules.indexer import load_vector_db
from configurations.config import INDEX_SOURCE_URL
import sys
import termios
from scripts.evaluator import enumerate_top_documents
import os
import json
from configurations.config import NQ_SAMPLE_SIZE
from matrics.results_logger import ResultsLogger, plot_score_distribution
import torch
from utility.logger import logger  # Import logger from utility/logger.py

tokenizer, model = load_model()


def profile_gpu():
    """
    Profiles GPU utilization and memory usage.
    """
    if torch.cuda.is_available():
        print("\n> GPU Utilization:")
        print(torch.cuda.memory_summary(device="cuda"))
    else:
        print("\n> No GPU detected.")


def check_device():
    """
    Checks and prints the available device for PyTorch (MPS, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        print("MPS backend is available. Using MPS for acceleration.")
        return "mps"
    elif torch.cuda.is_available():
        print("CUDA backend is available. Using CUDA for acceleration.")
        return "cuda"
    else:
        print("No GPU detected. Using CPU.")
        return "cpu"


# Example usage:
device = check_device()


def flush_input():
    """
    Flushes accidental keyboard input from the terminal buffer before reading input.

    This function ensures that any unintended input in the terminal buffer is cleared
    to avoid interference with subsequent user input.

    Raises:
        Exception: If an error occurs during the flush operation.
    """
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception as e:
        print(f"\n> Error received during input flush: {e} ")


def run_query_evaluation():
    logger.info("Starting query evaluation...")
    profile_gpu()

    logger.info("Loading external Vector DB...")
    vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

    run_mode = input("\n🛠️ Run in enumeration mode? (y/n): ").strip().lower()
    if run_mode == "y":
        from scripts.evaluator import hill_climb_documents

        mode_choice = input("\n🧪 Select mode: (e)numeration / (h)ill climbing: ").strip().lower()
        results_logger = ResultsLogger(top_k=5, mode="hill" if mode_choice == "h" else "enum")
        nq_file_path = "data/user_query_datasets/natural-questions-master/nq_open/NQ-open.dev.jsonl"

        if not os.path.exists(nq_file_path):
            logger.error(f"NQ file not found at: {nq_file_path}")
            return

        with open(nq_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= NQ_SAMPLE_SIZE:
                    break
                data = json.loads(line)
                query = data.get("question")
                logger.info(f"Running for NQ Query #{i + 1}: {query}")

                if mode_choice == "h":
                    result = hill_climb_documents(i, NQ_SAMPLE_SIZE, query, vector_db, model, tokenizer,
                                                  embedding_model, top_k=5)
                    results_logger.log(result)
                elif mode_choice == "e":
                    result = enumerate_top_documents(i, NQ_SAMPLE_SIZE, query, vector_db, embedding_model, top_k=5)
                    results_logger.log(result)

        return
    else:
        logger.info("Entering interactive query mode...")
        while True:
            flush_input()
            user_prompt = input("\n💬 Enter your query: ")
            if user_prompt.lower() == "exit":
                logger.info("Exiting application.")
                break

            result = query_model(user_prompt, model, tokenizer, device, vector_db, embedding_model, max_retries=3,
                                 quality_threshold=0.5)
            if result["error"]:
                logger.error(f"Error: {result['error']}")
            else:
                logger.info(f"Question: {result['question']}, Answer: {result['answer'].strip()}")


def run_analysis():
    """
    Runs analysis on logged results, summarizing scores and plotting score distributions.

    This function is triggered when the script is run with the `--analyze` argument.
    """
    logger.info("Running analysis on logged results...")
    logger_instance = ResultsLogger()
    logger_instance.summarize_scores()  # Print average, min, max
    plot_score_distribution()  # Show histogram of score distribution


if __name__ == "__main__":
    logger.info("Application started.")
    if "--analyze" in sys.argv:
        run_analysis()
    else:
        run_query_evaluation()


# === File: configurations/config.py ===
# config.py - FIXED VERSION
# ✅ This replaces your current config.py

# === MODEL CONFIGURATION ===
# ✅ Use a proper text generation model instead of DistilBERT
MODEL_PATH = "microsoft/DialoGPT-small"  # Good for conversations, lightweight
# Alternative options (uncomment one if you prefer):
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

# === File: types/config_enhanced.py ===
import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Centralized model configuration"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-medium"  # Better for conversations
    device_priority: list = None

    def __post_init__(self):
        if self.device_priority is None:
            self.device_priority = ["cuda", "mps", "cpu"]


@dataclass
class RetrievalConfig:
    """RAG-specific configuration"""
    top_k: int = 5
    similarity_cutoff: float = 0.75
    max_context_length: int = 4000
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class OptimizationConfig:
    """Hill climbing and optimization settings"""
    max_retries: int = 3
    quality_threshold: float = 0.7
    temperature: float = 0.7
    max_new_tokens: int = 128
    convergence_threshold: float = 0.01

# === File: classes/EmbeddingVectorStoreIndex.py ===
from llama_index.core import VectorStoreIndex
from typing import List, Tuple, Dict, Callable
import numpy as np


class EmbeddingVectorStoreIndex:
    def __init__(self, index: VectorStoreIndex, embed_fn: Callable[[str], np.ndarray]):
        """
        A wrapper around VectorStoreIndex that allows access to document embeddings.

        Args:
            index (VectorStoreIndex): The original LlamaIndex vector index.
            embed_fn (Callable[[str], np.ndarray]): Function that computes the embedding from text.
        """
        self.index = index
        self.embed_fn = embed_fn
        self.embeddings: Dict[str, np.ndarray] = {}

        # Extract and store embeddings from all documents in the docstore
        all_node_ids = list(index.docstore.docs.keys())
        for node in index.docstore.get_nodes(all_node_ids):
            if node.text:
                try:
                    embedding = embed_fn(node.text)
                    self.embeddings[node.node_id] = np.array(embedding)
                except Exception as e:
                    print(f"Failed to embed node {node.node_id[:6]}: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, np.ndarray]]:
        """
        Retrieve top-k nodes using the vector index, along with their scores and precomputed embeddings.

        Returns:
            List of (text, score, embedding) tuples.
        """
        nodes = self.index.as_retriever(similarity_top_k=top_k).retrieve(query)
        return [
            (node.node.text, node.score, self.embeddings.get(node.node.node_id))
            for node in nodes if node.node.node_id in self.embeddings
        ]

    def get_embedding(self, node_id: str) -> np.ndarray:
        """
        Get the embedding for a specific node/document ID.
        """
        return self.embeddings[node_id]


# === File: matrics/analyze_results.py ===
from results_logger import ResultsLogger, plot_score_distribution


def main():
    logger = ResultsLogger()

    print("📊 Summarizing scores from logged results...\n")
    logger.summarize_scores()

    print("\n📈 Displaying histogram of score distribution...\n")
    plot_score_distribution()


if __name__ == "__main__":
    main()


# === File: matrics/results_logger.py ===
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime


class ResultsLogger:
    def __init__(self, filepath=None, truncate_fields=None, max_length=500, top_k=None, mode=None):
        """
        Initialize the logger. If filepath is not given, generate one using timestamp, top_k, and mode.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["results"]

        if mode:
            filename_parts.append(mode)
        if top_k:
            filename_parts.append(f"top{top_k}")
        filename_parts.append(timestamp)

        auto_filename = "_".join(filename_parts) + ".jsonl"
        self.filepath = filepath or os.path.join("results", auto_filename)
        self.max_length = max_length
        self.truncate_fields = truncate_fields or {"query", "answer", "context"}

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def _truncate(self, value):
        if isinstance(value, str) and len(value) > self.max_length:
            return value[:self.max_length] + "...[truncated]"
        return value

    def log(self, record: dict):
        safe_record = {
            k: self._truncate(v) if k in self.truncate_fields else v
            for k, v in record.items()
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

    def load_results(self):
        """Load all logged results from the output file."""
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def summarize_scores(self):
        """Prints basic statistics (count, average, min, max) for similarity scores."""
        results = self.load_results()
        scores = [r["score"] for r in results if "score" in r and isinstance(r["score"], (int, float))]
        if not scores:
            print("No scores to summarize.")
            return

        print(f"\n📊 Results Summary:")
        print(f"Total entries with score: {len(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.4f}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")


# --- Histogram plotting function ---
def plot_score_distribution(filepath="results/outputs.jsonl"):
    """Plot a histogram of similarity scores from the results file."""
    if not os.path.exists(filepath):
        print(f"No results file found at {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        scores = [
            json.loads(line.strip()).get("score")
            for line in f if line.strip()
        ]

    scores = [s for s in scores if isinstance(s, (int, float))]

    if not scores:
        print("No valid scores found to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=10, color="skyblue", edgecolor="black")
    plt.title("Similarity Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === File: scripts/create_public_corpus.py ===
# create_public_corpus.py
import os
from datasets import load_dataset
from configurations.config import PUBLIC_CORPUS_DATASET, PUBLIC_CORPUS_DIR

# Load a small public corpus from the "wikitext" dataset
dataset = load_dataset("wikitext", PUBLIC_CORPUS_DATASET, split="train")

# Directory where we'll store the text files
data_dir = '../' + PUBLIC_CORPUS_DIR
os.makedirs(data_dir, exist_ok=True)

# Save each non-empty document to a separate text file
for i, example in enumerate(dataset):
    text = example["text"]
    if text.strip():  # only write non-empty documents
        with open(os.path.join(data_dir, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    print(f">Saved {i + 1} documents to '{data_dir}'")


# === File: scripts/dir_tree_printer.py ===
# dir_tree_printer.py
import os


def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def print_tree_summary(startpath, max_files_per_dir=5, depth=0):
    if not os.path.isdir(startpath):
        return

    entries = list(os.scandir(startpath))
    files = sorted([entry.name for entry in entries if entry.is_file()])
    dirs = sorted([entry for entry in entries if entry.is_dir()], key=lambda d: d.name)

    indent = '    ' * depth
    total_size = sum(os.path.getsize(os.path.join(startpath, f)) for f in files)
    print(f"{indent}{os.path.basename(startpath)}/ ({len(files)} files, {human_readable_size(total_size)})")

    for f in files[:max_files_per_dir]:
        display_name = f if len(f) <= 30 else f[:30] + "..."
        print(f"{indent}    {display_name}")
    if len(files) > max_files_per_dir:
        print(f"{indent}    ... +{len(files) - max_files_per_dir} more files")

    for d in dirs:
        print_tree_summary(d.path, max_files_per_dir, depth + 1)


print_tree_summary('../', max_files_per_dir=20)


# === File: scripts/evaluator.py ===
# evaluator.py
from typing import Tuple, Union
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from configurations.config import HF_MODEL_NAME, LLM_MODEL_NAME
from modules.query import retrieve_context
from utility.logger import logger


def load_llm() -> Tuple[PreTrainedTokenizer, Union[PreTrainedModel, AutoModelForCausalLM]]:
    """
    Load the LLM model and tokenizer.

    Returns:
        Tuple[PreTrainedTokenizer, Union[PreTrainedModel, AutoModelForCausalLM]]:
        A tuple containing the tokenizer and the model.
    """
    logger.info("Loading LLM model and tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    try:
        model: Union[PreTrainedModel, AutoModelForCausalLM] = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        logger.info(f"Loaded causal language model: {LLM_MODEL_NAME}")
    except ValueError:
        model = AutoModel.from_pretrained(LLM_MODEL_NAME)
        logger.info(f"Loaded masked language model: {LLM_MODEL_NAME}")

    return tokenizer, model


def load_embedding_model():
    """
    Load the embedding model.

    Returns:
        SentenceTransformer: The embedding model instance.
    """
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(HF_MODEL_NAME)
    logger.info(f"Loaded embedding model: {HF_MODEL_NAME}")
    return embedding_model


def run_llm_query(query: str, tokenizer: PreTrainedTokenizer,
                  model: Union[PreTrainedModel, AutoModelForCausalLM]) -> str:
    """
    Run a query using the LLM.

    Args:
        query (str): The input query.
        tokenizer (PreTrainedTokenizer): The tokenizer for the LLM.
        model (Union[PreTrainedModel, AutoModelForCausalLM]): The LLM model.

    Returns:
        str: The result of the query.
    """
    logger.info(f"Running query: {query}")
    inputs = tokenizer(query, return_tensors="pt")

    if hasattr(model, "generate"):
        outputs = model.generate(**inputs, max_new_tokens=100)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated result: {result}")
    else:
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        result = str(embeddings)
        logger.info(f"Computed embeddings: {result}")

    return result


def sanity_check(user_query, optimized_user_query, vector_db, vector_query=None):
    """
    Perform a sanity check for query optimization.

    Args:
        user_query (str): The original user query.
        optimized_user_query (str): The optimized user query.
        vector_db: The vector database instance.
        vector_query: Optional vector representation of the query.

    Returns:
        dict: Results of the sanity check.
    """
    print(f"\n> Running sanity check for query optimization...")

    tokenizer, model = load_llm()

    orig_answer = run_llm_query(user_query, tokenizer, model) if vector_query is None else retrieve_context(
        query=vector_query if not convert_to_vector else user_query,
        vector_db=vector_db,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2"),
    )
    opt_answer = run_llm_query(optimized_user_query, tokenizer, model) if vector_query is None else retrieve_context(
        query=vector_query if not convert_to_vector else optimized_user_query,
        vector_db=vector_db,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2"),
    )

    embedding_judgment = compare_answers_with_embeddings(user_query, orig_answer, opt_answer)
    llm_judgment = judge_with_llm(user_query, orig_answer, opt_answer, model=model, tokenizer=tokenizer)

    print("\n🔍 **Sanity Check Results**:")
    print(f"📝 Original Query: {user_query}")
    print(f"📝 Optimized Query: {optimized_user_query}")
    print(f"💬 Original Answer: {orig_answer}")
    print(f"💬 Optimized Answer: {opt_answer}")
    print(f"📊 LLM Judgment: {llm_judgment}")
    print(f"📊 Embedding Judgment: {embedding_judgment}")
    print(
        f"⚖ Final Decision: {'Optimized' if llm_judgment == 'Optimized' or embedding_judgment == 'Optimized' else 'Original'}")

    return {
        "original_query": user_query,
        "optimized_query": optimized_user_query,
        "original_answer": orig_answer,
        "optimized_answer": opt_answer,
        "llm_judgment": llm_judgment,
        "embedding_judgment": embedding_judgment,
        "final_decision": "Optimized" if llm_judgment == "Optimized" or embedding_judgment == "Optimized" else "Original"
    }


def enumerate_top_documents(i, num, query, index, embedding_model, top_k=5, convert_to_vector=False):
    """
    Enumerate top documents for a given query using embedding-based retrieval.

    Args:
        i (int): Current query index.
        num (int): Total number of queries.
        query (str): The input query.
        index: The document index instance.
        embedding_model: The embedding model used for retrieval.
        top_k (int): Number of top documents to retrieve.
        convert_to_vector (bool): Whether to convert the query to a vector.

    Returns:
        dict: Results containing the query and top documents.
    """
    logger.info(f"Enumerating top documents for query #{i + 1} of {num}: {query}")
    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    query_vector = get_text_embedding(query, embedding_model) if convert_to_vector else query
    results = retriever.retrieve(query_vector)  # Retrieve top documents

    top_docs = []
    for rank, node_with_score in enumerate(results, start=1):
        score = node_with_score.score if hasattr(node_with_score, "score") else None
        content = node_with_score.node.get_content()
        top_docs.append({
            "rank": rank,
            "score": score,
            "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
        })

    result = {
        "query": query,
        "top_documents": top_docs
    }
    logger.info(f"Top documents enumerated: {result}")
    return result


def hill_climb_documents(i, num, query, index, llm_model, tokenizer, embedding_model, top_k=5, max_tokens=100,
                         convert_to_vector=False):
    """
    Perform hill climbing to find the best answer for a query.

    Args:
        i (int): Current query index.
        num (int): Total number of queries.
        query (str): The input query.
        index: The document index instance.
        llm_model: The language model used for generating answers.
        tokenizer: The tokenizer for the language model.
        embedding_model: The embedding model used for similarity calculations.
        top_k (int): Number of top documents to retrieve.
        max_tokens (int): Maximum tokens for LLM-generated answers.
        convert_to_vector (bool): Whether to convert the query to a vector.

    Returns:
        dict: Results containing the query, best answer, and context.
    """
    logger.info(f"Starting hill climbing for query #{i + 1} of {num}: {query}")
    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    query_vector = get_text_embedding(query, embedding_model) if convert_to_vector else query
    results = retriever.retrieve(query_vector)  # Retrieve top documents

    best_score = -1.0
    best_answer = None
    best_context = None

    contexts = [node_with_score.node.get_content() for node_with_score in results]
    if not contexts:
        logger.warning("No contexts retrieved.")
        return {"query": query, "answer": None, "context": None}

    prompts = [f"Context:\n{context}\n\nQuestion: {query}\nAnswer:" for context in contexts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    outputs = llm_model.generate(**inputs, max_new_tokens=max_tokens)
    answers = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

    for context, answer in zip(contexts, answers):
        if not answer:  # Skip empty answers
            continue

        query_emb = get_text_embedding(query, embedding_model)
        answer_emb = get_text_embedding(answer, embedding_model)
        score = calculate_cosine_similarity(query_emb, answer_emb)  # Calculate similarity score

        if score > best_score:
            best_score = score
            best_answer = answer
            best_context = context

    logger.info(f"Best answer selected with score {best_score}: {best_answer}")
    return {"query": query, "answer": best_answer, "context": best_context}


def compare_answers_with_embeddings(user_query, original_answer, optimized_answer):
    """
    Compare answers using embeddings and cosine similarity.

    Args:
        user_query (str): The original user query.
        original_answer (str): The original answer to compare.
        optimized_answer (str): The optimized answer to compare.

    Returns:
        str: "Optimized", "Original", or "Tie" based on the similarity scores.
    """
    logger.info("Comparing answers with embeddings.")
    embedding_model = load_embedding_model()  # Load the embedding model

    # Generate embeddings for the query and answers
    query_embedding = get_text_embedding(user_query, embedding_model)
    orig_embedding = get_text_embedding(original_answer, embedding_model)
    opt_embedding = get_text_embedding(optimized_answer, embedding_model)

    # Calculate cosine similarity scores
    sim_orig = calculate_cosine_similarity(query_embedding, orig_embedding)
    sim_opt = calculate_cosine_similarity(query_embedding, opt_embedding)

    logger.info(f"Similarity scores - Original: {sim_orig}, Optimized: {sim_opt}")
    return "Optimized" if sim_opt > sim_orig else "Original" if sim_orig > sim_opt else "Tie"


def judge_with_llm(user_query, original_answer, optimized_answer, model=None, tokenizer=None, device=None):
    """
    Judge answers using a language model (LLM).

    Args:
        user_query (str): The original user query.
        original_answer (str): The original answer to compare.
        optimized_answer (str): The optimized answer to compare.
        model: The LLM model instance (optional).
        tokenizer: The tokenizer for the LLM (optional).
        device (str): The device to run the model on (optional).

    Returns:
        str: "Optimized", "Original", or "Tie" based on the LLM's judgment.
    """
    logger.info("Judging answers with LLM.")
    if model is None or tokenizer is None:
        tokenizer, model = load_llm()  # Load the LLM and tokenizer

    # Determine the device to use
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare the prompt for the LLM
    prompt = f"""You are an AI judge evaluating query optimization. Compare the two answers below and choose the best one.

        Query: {user_query}

        Original Answer: {original_answer}

        Optimized Answer: {optimized_answer}

        Answer ONLY with exactly one word: "Optimized", "Original", or "Tie". Do not include any extra text.
        """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output using the LLM
    with torch.cuda.amp.autocast(enabled=device == "cuda"):
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    logger.info(f"LLM judgment result: {answer}")
    for option in ["Optimized", "Original", "Tie"]:
        if option.lower() in answer.lower():
            return option
    return answer


# === File: scripts/modelHuggingFaceDownload.py ===
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

# === File: modules/query.py ===
# query.py
import numpy as np
from typing import Union, Optional, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import heapq
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from configurations.config import MAX_RETRIES, QUALITY_THRESHOLD, MAX_NEW_TOKENS
from utility.embedding_utils import get_query_vector
from utility.logger import logger  # Import logger


def vector_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    This function computes the cosine similarity score, which is a measure of similarity between two non-zero vectors.

    In the VectorStoreIndex, the similarity function is : similarity = np.dot(vec1, vec2) / (||vec1|| * ||vec2||)
    Args:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between the two vectors.
    """
    logger.info("Calculating cosine similarity between two vectors.")
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0.0 or norm2 == 0.0:
        logger.warning(
            f"One or both vectors are zero vectors, returning similarity as 0.0. vector1: {vector1} vector2: {vector2}")
        return 0.0  # or float('nan') depending on use case
    similarity = np.dot(vector1, vector2) / (norm1 * norm2)
    logger.info(f"Cosine similarity calculated: {similarity}")
    return similarity


def get_llamaindex_compatible_text(node) -> str:
    """
    Extracts text from a node in the exact same way LlamaIndex does for embedding.

    Args:
        node: LlamaIndex node object

    Returns:
        str: Text processed exactly as LlamaIndex processes it for embedding
    """
    try:
        # Use the same method LlamaIndex uses for embedding
        return node.get_content(metadata_mode=MetadataMode.EMBED)
    except Exception as e:
        logger.warning(f"Failed to get LlamaIndex-compatible text: {e}. Falling back to node.text")
        return node.text


def get_cached_embedding_llamaindex_style(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Retrieves the cached embedding for a given text using LlamaIndex's exact method.

    This function replicates exactly how LlamaIndex creates embeddings for documents.

    Args:
        text (str): Text to convert into an embedding (should be the full processed text with metadata).
        embed_model (HuggingFaceEmbedding): Embedding model to use.

    Returns:
        np.ndarray: Cached embedding vector for the text.
    """
    logger.info(f"Retrieving cached embedding for text: {text[:50]}...")

    # Use get_text_embedding (not get_query_embedding) to match document encoding
    embedding = embed_model.get_text_embedding(text)
    embedding_array = np.array(embedding)

    # Check if the model returns normalized vectors
    norm = np.linalg.norm(embedding_array)
    if norm > 1.1 or norm < 0.9:  # Not normalized
        embedding_array = embedding_array / norm
        logger.debug("Applied manual normalization to embedding")

    logger.info("Cached embedding retrieved successfully.")
    return embedding_array


def retrieve_context_aligned_to_llama_index(
        query: Union[str, np.ndarray],
        vector_db: VectorStoreIndex,
        embed_model: HuggingFaceEmbedding,
        top_k: int = 5,
        similarity_cutoff: float = 0.5,
) -> str:
    """
    Retrieves relevant context from the vector database using LlamaIndex-compatible embeddings.

    Args:
        query (Union[str, np.ndarray]): Query text or vector.
        vector_db (VectorStoreIndex): Vector database for document retrieval.
        embed_model (HuggingFaceEmbedding): Embedding model for vector conversion.
        top_k (int): Number of top results to retrieve.
        similarity_cutoff (float): Minimum similarity score to include results.

    Returns:
        str: Retrieved context as a concatenated string.
    """
    logger.info("Retrieving context for the query with LlamaIndex compatibility.")
    if not isinstance(query, (str, np.ndarray)):
        logger.error("Query must be a string or a numpy.ndarray.")
        raise TypeError("Query must be a string or a numpy.ndarray.")

    # Use LlamaIndex's native retriever first
    retriever = vector_db.as_retriever(similarity_top_k=top_k)
    nodes_with_scores = retriever.retrieve(query)

    if not nodes_with_scores:
        logger.warning("No nodes retrieved from the vector DB.")
        return ''

    logger.info(f"Retrieved {len(nodes_with_scores)} nodes from vector database.")

    # Get query vector
    query_vector: Optional[np.ndarray] = None
    if isinstance(query, str):
        if embed_model is None:
            logger.error("embed_model is required for converting string queries to vectors.")
            raise ValueError("embed_model is required for converting string queries to vectors.")
        query_vector = get_query_vector(query, embed_model)
    elif isinstance(query, np.ndarray):
        query_vector = query

    # Extract nodes and their LlamaIndex-compatible text
    nodes = [node_with_score.node for node_with_score in nodes_with_scores]
    llamaindex_texts = [get_llamaindex_compatible_text(node) for node in nodes]

    # Get embeddings using LlamaIndex's exact method
    document_embeddings = np.array([
        get_cached_embedding_llamaindex_style(text, embed_model)
        for text in llamaindex_texts
    ])

    # Get LlamaIndex's original scores for comparison
    llamaindex_scores = [node_with_score.score for node_with_score in nodes_with_scores]

    # Calculate similarities manually to verify/compare
    if query_vector is not None and document_embeddings is not None:
        # Manual similarity calculation (should match LlamaIndex's results)
        manual_similarities = np.dot(document_embeddings, query_vector) / (
                np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_vector)
        )

        logger.info("Similarity comparison:")
        for i, (manual_sim, llamaindex_sim) in enumerate(zip(manual_similarities, llamaindex_scores)):
            diff = abs(manual_sim - llamaindex_sim)
            logger.info(f"Node {i}: Manual={manual_sim:.6f}, LlamaIndex={llamaindex_sim:.6f}, Diff={diff:.6f}")

        # Use manual similarities for filtering (they should match LlamaIndex's)
        similarity_scores = manual_similarities
    else:
        logger.warning("Query vector or document embeddings are None, using LlamaIndex scores.")
        similarity_scores = llamaindex_scores

    # Get the actual content for each node (not the metadata-enhanced text)
    node_contents = [node.get_content() for node in nodes]

    # Create scored pairs and filter
    scored_nodes = list(zip(similarity_scores, node_contents))
    top_nodes = heapq.nlargest(top_k, scored_nodes, key=lambda x: x[0])
    filtered_nodes = [content for score, content in top_nodes if score >= similarity_cutoff]

    logger.info(f"Filtered {len(filtered_nodes)} nodes based on similarity cutoff.")
    return "\n".join(filtered_nodes)


# Keep the original function as fallback
def retrieve_context(
        query: Union[str, np.ndarray],
        vector_db: VectorStoreIndex,
        embed_model: HuggingFaceEmbedding,
        top_k: int = 5,
        similarity_cutoff: float = 0.5,
) -> str:
    """
    Original retrieve_context function - kept for backward compatibility.
    Use retrieve_context_improved for better LlamaIndex compatibility.
    """
    logger.warning(
        "Using original retrieve_context. Consider using retrieve_context_improved for better compatibility.")
    return retrieve_context_aligned_to_llama_index(query, vector_db, embed_model, top_k, similarity_cutoff)


def evaluate_answer_quality(
        answer: str,
        question: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> float:
    """
    Evaluates the quality of the generated answer based on the given question.

    Args:
        answer (str): Generated answer.
        question (str): Original question.
        model (AutoModelForCausalLM): Language model used for evaluation.
        tokenizer (GPT2TokenizerFast): Tokenizer for processing text.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Quality score of the answer.
    """
    logger.info("Evaluating the quality of the answer.")
    # Placeholder for evaluation logic
    score = 1.0
    logger.info(f"Quality score calculated: {score}")
    return score


def rephrase_query(
        original_prompt: str,
        previous_answer: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> str:
    """
    Rephrases the original query based on the previous answer to improve the response.

    Args:
        original_prompt (str): Original query prompt.
        previous_answer (str): Previous answer generated by the model.
        model (AutoModelForCausalLM): Language model used for rephrasing.
        tokenizer (GPT2TokenizerFast): Tokenizer for processing text.
        device (torch.device): Device to run the rephrasing on.

    Returns:
        str: Rephrased query.
    """
    logger.info("Rephrasing the query based on the previous answer.")
    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"The previous answer was: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )
    inputs = tokenizer(rephrase_prompt, return_tensors="pt", truncation=True, max_length=1020).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7
    )
    rephrased_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    logger.info("Query rephrased successfully.")
    return rephrased_query


def query_model(
        prompt: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device,
        vector_db: Optional[VectorStoreIndex] = None,
        embedding_model: Optional[HuggingFaceEmbedding] = None,
        max_retries: int = MAX_RETRIES,
        quality_threshold: float = QUALITY_THRESHOLD,
        use_improved_retrieval: bool = True
) -> Dict[str, Union[str, float, int, None]]:
    """
    Queries the language model with the given prompt and retrieves an answer.

    Args:
        prompt (str): Query prompt.
        model (AutoModelForCausalLM): Language model to generate the answer.
        tokenizer (GPT2TokenizerFast): Tokenizer for processing text.
        device (torch.device): Device to run the query on.
        vector_db (Optional[VectorStoreIndex]): Vector database for context retrieval.
        embedding_model (Optional[HuggingFaceEmbedding]): Embedding model for vector conversion.
        max_retries (int): Maximum number of retries for improving the answer.
        quality_threshold (float): Minimum quality score to accept the answer.
        use_improved_retrieval (bool): Whether to use the improved LlamaIndex-compatible retrieval.

    Returns:
        Dict[str, Union[str, float, int, None]]: Dictionary containing the query result.
    """
    logger.info("Starting query process with the model.")
    try:
        answer: str = ""
        score: float = 0.0
        attempt: int = 0
        current_prompt: str = prompt

        while attempt <= max_retries:
            try:
                if vector_db is not None:
                    if use_improved_retrieval:
                        retrieved_context = retrieve_context_aligned_to_llama_index(current_prompt, vector_db,
                                                                                    embedding_model)
                    else:
                        retrieved_context = retrieve_context(current_prompt, vector_db, embedding_model)
                    augmented_prompt = f"Context: {retrieved_context}\n\nQuestion: {current_prompt}\nAnswer:"
                else:
                    augmented_prompt = current_prompt

                inputs = tokenizer(augmented_prompt, return_tensors="pt", truncation=True, max_length=1020 ).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                score = evaluate_answer_quality(answer, current_prompt, model, tokenizer, device)

                if score >= quality_threshold or attempt == max_retries:
                    logger.info("Query process completed successfully.")
                    return {
                        "question": prompt,
                        "answer": answer,
                        "score": score,
                        "attempts": attempt + 1,
                        "error": None
                    }

                current_prompt = rephrase_query(current_prompt, answer, model, tokenizer, device)
                attempt += 1
            except ValueError as err:
                logger.error(f"Error during query process: {err}")
                return {
                    "question": prompt,
                    "answer": answer,
                    "score": score,
                    "attempts": attempt + 1,
                    "error": str(err)
                }

        if not answer.strip():
            logger.error("Model did not generate a response.")
            return {
                "question": prompt,
                "answer": "Model did not generate a response.",
                "score": 0.0,
                "attempts": attempt + 1,
                "error": "Empty response from model"
            }

        logger.info("Query process completed successfully.")
        return {
            "final_answer": answer,
            "final_score": score,
            "attempts": attempt + 1,
            "final_prompt": current_prompt
        }
    except ValueError as err:
        logger.error(f"Error during query process: {err}")
        return {
            "question": prompt,
            "answer": "Error during generation.",
            "score": 0.0,
            "attempts": 0,
            "error": str(err)
        }


# === File: modules/indexer.py ===
# indexer.py
import os
from urllib.parse import urlparse
import requests
from datasets import load_dataset
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from configurations.config import DATA_PATH, HF_MODEL_NAME
from utility.logger import logger
from typing import Tuple, Optional


def parse_source_path(source_path: str) -> Tuple[str, str]:
    """
    Parses the source path to determine its type and extract relevant information.

    Args:
        source_path (str): The source path to parse.

    Returns:
        Tuple[str, str]: A tuple containing the source type ('url' or 'hf') and the parsed corpus or dataset configuration.

    Raises:
        ValueError: If the source path format is unsupported.
    """
    logger.info(f"Parsing source path: {source_path}")
    if source_path.startswith("http://") or source_path.startswith("https://"):
        corpus_name = source_path.split("/")[-1]
        logger.info(f"Source type identified as URL with corpus name: {corpus_name}")
        return "url", corpus_name
    elif ":" in source_path:
        dataset_config = source_path.replace(":", "_")
        logger.info(f"Source type identified as Hugging Face dataset with config: {dataset_config}")
        return "hf", dataset_config
    else:
        logger.error(f"Unsupported source path format: {source_path}")
        raise ValueError(f"Unsupported source path format: {source_path}")


def download_and_save_from_hf(dataset_name: str, config: str, target_dir: str, max_docs: int = 1000) -> None:
    """
    Downloads a dataset from Hugging Face and saves it locally as text files in batches.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        config (str): Configuration for the dataset.
        target_dir (str): Directory to save the downloaded documents.
        max_docs (int): Maximum number of documents to download.

    Returns:
        None
    """
    logger.info(f"Downloading dataset '{dataset_name}' with config '{config}' from Hugging Face...")
    dataset = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
    os.makedirs(target_dir, exist_ok=True)

    batch_size = 100
    batch = []
    for i, item in enumerate(dataset):
        if i >= max_docs:
            break
        text = item["text"].strip()
        if text:
            batch.append((i, text))
        if len(batch) == batch_size or i == max_docs - 1:
            for doc_id, doc_text in batch:
                with open(os.path.join(target_dir, f"doc_{doc_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(doc_text)
            batch.clear()
    logger.info(f"Saved {min(max_docs, len(dataset))} documents to {target_dir}")


def validate_url(url: str) -> bool:
    """
    Validates the URL to ensure it is safe and matches expected patterns.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.

    Raises:
        ValueError: If the URL is invalid or unsafe.
    """
    logger.info(f"Validating URL: {url}")
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        logger.error("URL must use HTTPS.")
        raise ValueError("URL must use HTTPS.")
    allowed_domains = ["example.com", "trusted-source.com"]
    if parsed_url.netloc not in allowed_domains:
        logger.error(f"Domain '{parsed_url.netloc}' is not allowed.")
        raise ValueError(f"Domain '{parsed_url.netloc}' is not allowed.")
    return True


def download_and_save_from_url(url: str, target_dir: str) -> None:
    """
    Downloads text data from a URL and saves it locally using streaming.

    Args:
        url (str): URL to download the data from.
        target_dir (str): Directory to save the downloaded data.

    Returns:
        None
    """
    validate_url(url)
    logger.info(f"Downloading data from URL: {url}")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, "corpus.txt")

    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            logger.error(f"Failed to download from {url}, status code: {response.status_code}")
            raise Exception(f"Failed to download from {url}, status code: {response.status_code}")
        with open(file_path, "w", encoding="utf-8") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk.decode("utf-8"))
    logger.info(f"Downloaded and saved corpus to {file_path}")


def load_vector_db(source: str = "local", source_path: Optional[str] = None) -> (
        VectorStoreIndex, HuggingFaceEmbedding):
    """
    Loads or creates a vector database for document retrieval with optimized embedding model caching.

    Args:
        source (str): Source type ('local' or 'url').
        source_path (Optional[str]): Path to the data source.

    Returns:
        GPTVectorStoreIndex: Loaded or newly created vector database.
    """
    logger.info(f"Loading vector database from source: {source}, source_path: {source_path}")
    embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    if not hasattr(load_vector_db, "_embed_model"):
        load_vector_db._embed_model = embedding_model
    if source == "url":
        if source_path is None:
            logger.error("source_path must be provided for 'url' source.")
            raise ValueError(
                "source_path must be provided for 'url' source. Please insert into config.py a INDEX_SOURCE_URL "
                "variable with valid data")
        source_type, corpus_name = parse_source_path(source_path)
        corpus_dir = os.path.join("data", corpus_name)
        storage_dir = os.path.join("storage", corpus_name)

        if not os.path.exists(corpus_dir):
            logger.info(f"Downloading corpus into {corpus_dir}...")
            if source_type == "hf":
                dataset, config = source_path.split(":", 1)
                download_and_save_from_hf(dataset, config, corpus_dir)
            else:
                download_and_save_from_url(source_path, corpus_dir)

        if os.path.exists(storage_dir):
            logger.info(f"Loading existing vector database from {storage_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            vector_db = load_index_from_storage(storage_context, embed_model=embedding_model)
            logger.info(f"Loaded existing vector database for '{corpus_name}' from {storage_dir}.")
            return vector_db, embedding_model

        else:
            logger.info(f"Indexing documents from {corpus_dir}...")
            documents = SimpleDirectoryReader(corpus_dir).load_data()
            vector_db = GPTVectorStoreIndex.from_documents(
                documents,
                store_nodes_override=True,
                embed_model=embedding_model
            )
            vector_db.storage_context.persist(persist_dir=storage_dir)
            logger.info(f"Indexed {len(documents)} documents and saved to {storage_dir}.")
            return vector_db, embedding_model

    else:
        storage_dir = "storage"
        if os.path.exists(storage_dir):
            logger.info("Loading existing local vector database from 'storage/'.")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            vector_db = load_index_from_storage(storage_context, embed_model=embedding_model)
            logger.info("Loaded existing local vector database from 'storage/'.")
            return vector_db, embedding_model

        logger.info("Indexing documents from local data path...")
        documents = SimpleDirectoryReader(DATA_PATH).load_data()
        vector_db = GPTVectorStoreIndex.from_documents(documents, embed_model=embedding_model)
        vector_db.storage_context.persist(persist_dir=storage_dir)
        logger.info("Indexed and saved new local corpus to 'storage/'.")
        return vector_db, embedding_model


# === File: modules/model_loader.py ===
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
import torch
from configurations.config import MODEL_PATH
from typing import Tuple
from utility.logger import logger


def load_model() -> Tuple[AutoTokenizer, torch.nn.Module]:
    logger.info(f"Loading Model {MODEL_PATH}...")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Add padding token if it doesn't exist (common issue with DialoGPT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added padding token to tokenizer.")

    logger.info("Tokenizer loaded successfully.")

    # Try to load as CausalLM first (for text generation)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        logger.info("✅ Loaded AutoModelForCausalLM (supports text generation).")

        # Apply optimizations
        if torch.__version__.startswith("2"):
            try:
                model = torch.compile(model)
                logger.info("Model optimized using torch.compile.")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return tokenizer, model

    except Exception as e:
        logger.warning(f"Failed to load as CausalLM: {e}")

        # Fallback: Try sequence classification
        try:
            config = AutoConfig.from_pretrained(MODEL_PATH)
            architectures = getattr(config, "architectures", [])

            if any("SequenceClassification" in arch for arch in architectures):
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                logger.info("✅ Loaded AutoModelForSequenceClassification.")
            else:
                # Last resort: generic model (won't support generation)
                model = AutoModel.from_pretrained(MODEL_PATH)
                logger.warning("⚠️ Loaded generic AutoModel (does NOT support text generation).")
                logger.warning("This model can only be used for embeddings/understanding, not generation.")

            return tokenizer, model

        except Exception as e2:
            logger.error(f"Failed to load model with any method: {e2}")
            raise e2


def get_model_capabilities(model) -> dict:
    """
    Check what the loaded model can do
    """
    capabilities = {
        'can_generate': hasattr(model, 'generate'),
        'model_type': type(model).__name__,
        'is_causal_lm': 'CausalLM' in type(model).__name__,
        'is_sequence_classification': 'SequenceClassification' in type(model).__name__,
    }

    return capabilities


def test_model_loading():
    """
    Test function to verify model loading works correctly
    """
    logger.info("Testing model loading...")

    try:
        tokenizer, model = load_model()
        capabilities = get_model_capabilities(model)

        logger.info("Model loading test results:")
        for key, value in capabilities.items():
            status = "✅" if value else "❌"
            logger.info(f"  {status} {key}: {value}")

        if capabilities['can_generate']:
            logger.info("🎉 Model supports text generation!")

            # Quick generation test
            test_input = tokenizer("Hello", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **test_input,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"  Test generation: '{result}'")

        else:
            logger.warning("⚠️ Model does NOT support text generation.")
            logger.warning("Consider switching to a different model in config.py")

        return capabilities['can_generate']

    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test when this file is executed directly
    test_model_loading()


# === File: utility/embedding_utils.py ===
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# from logger import logger

def get_query_vector(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Converts a query's text into a vector embedding using the specified embedding model.

    Args:
        text (str): The input query text.
        embed_model (HuggingFaceEmbedding): The embedding model used for generating embeddings.

    Returns:
        np.ndarray: The vector embedding of the query text.
    """
    # logger.info(f"Generating query vector for text: {text[:30]}...")
    vector = embed_model.get_query_embedding(text)
    # logger.info("Query vector generated successfully.")
    return np.array(vector, dtype=np.float32)


# === File: utility/logger.py ===
import logging

# Configure logger
logger = logging.getLogger("ModularRAGOptimization")
logger.setLevel(logging.INFO)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)

# File handler for logging to a file
file_handler = logging.FileHandler("logger.log")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


