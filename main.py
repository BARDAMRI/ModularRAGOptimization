from modules.model_loader import load_model
from modules.query import query_model
from modules.indexer import load_index
import torch
from config import DEVICE_TYPE_MPS, DEVICE_TYPE_CPU, INDEX_SOURCE_URL
import sys
import termios
from scripts.evaluator import enumerate_top_documents
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import json
from config import NQ_SAMPLE_SIZE
from matrics.results_logger import ResultsLogger, plot_score_distribution


# Flushes any accidental keyboard input from the terminal buffer before reading input
def flush_input():
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception as e:
        print(f"\n> Error received during input flush: {e} ")


def run_query_evaluation():
    # Load the tokenizer and LLM model (e.g., LLaMA) from safetensors
    tokenizer, model = load_model()
    device = DEVICE_TYPE_MPS if torch.backends.mps.is_available() else DEVICE_TYPE_CPU

    print("\n> Warming up the model")
    _ = model.generate(
        **tokenizer("Warmup", return_tensors="pt").to(device),
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id
    )

    print("\n> Loading external Vector DB ")
    index = load_index(source="url", source_path=INDEX_SOURCE_URL)

    run_mode = input("\n🛠️ Run in enumeration mode? (y/n): ").strip().lower()
    if run_mode == "y":
        from scripts.evaluator import hill_climb_documents

        mode_choice = input("\n🧪 Select mode: (e)numeration / (h)ill climbing: ").strip().lower()
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        logger = ResultsLogger(top_k=5, mode="hill" if mode_choice == "h" else "enum")
        nq_file_path = "data/user_query_datasets/natural-questions-master/nq_open/NQ-open.dev.jsonl"

        if not os.path.exists(nq_file_path):
            print(f"❌ NQ file not found at: {nq_file_path}")
            return

        with open(nq_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= NQ_SAMPLE_SIZE:
                    break
                data = json.loads(line)
                query = data.get("question")
                print(f"\n🔍 Running for NQ Query #{i + 1}: {query}")

                if mode_choice == "h":
                    result = hill_climb_documents(i, NQ_SAMPLE_SIZE, query, index, model, tokenizer, embedding_model, top_k=5)
                    logger.log(result)
                elif mode_choice == "e":
                    result = enumerate_top_documents(i, NQ_SAMPLE_SIZE, query, index, embedding_model, top_k=5)
                    logger.log(result)
            else:
                print("\n> Invalid input received...")
        return
    else:
        print("\n> Ask anything you want!\nType 'exit' to stop.")
        while True:
            flush_input()
            user_prompt = input("\n💬 Enter your query: ")
            if user_prompt.lower() == "exit":
                print("> Exiting. Have a great day!")
                break

            result = query_model(user_prompt, model, tokenizer, device, index)
            if result["error"]:
                print(f"\n⚠️ Error: {result['error']}")
            else:
                print(f"\nQuestion: {result['question']}\nAnswer: {result['answer'].strip()}")


def run_analysis():
    from matrics.results_logger import ResultsLogger, plot_score_distribution

    logger = ResultsLogger()
    logger.summarize_scores()  # Print average, min, max
    plot_score_distribution()  # Show histogram of score distribution


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        run_analysis()
    else:
        run_query_evaluation()
