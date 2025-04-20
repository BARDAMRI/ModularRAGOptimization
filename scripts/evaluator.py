# evaluator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from config import HF_MODEL_NAME
"""Load Models"""
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDING_MODEL_NAME = HF_MODEL_NAME


def load_llm():
    """Loads the LLM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model


def load_embedding_model():
    """Loads the sentence embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def run_llm_query(query):
    """Runs the LLM on a given query and returns the generated answer."""
    tokenizer, model = load_llm()
    # Use GPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def compare_answers_with_embeddings(original_query, original_answer, optimized_answer):
    """Uses sentence embeddings to compare how well each answer aligns with the original query."""
    model = load_embedding_model()

    # Convert query and answers to embeddings
    query_embedding = model.encode(original_query, convert_to_tensor=True)
    orig_embedding = model.encode(original_answer, convert_to_tensor=True)
    opt_embedding = model.encode(optimized_answer, convert_to_tensor=True)

    # Compute similarity scores
    sim_orig = util.pytorch_cos_sim(query_embedding, orig_embedding).item()
    sim_opt = util.pytorch_cos_sim(query_embedding, opt_embedding).item()

    # Compare how well each response aligns with the original query
    return "Optimized" if sim_opt > sim_orig else "Original" if sim_orig > sim_opt else "Tie"


def judge_with_llm(original_query, original_answer, optimized_answer):
    """Uses an LLM to decide which answer is better, returning only one word."""
    tokenizer, model = load_llm()

    prompt = f"""You are an AI judge evaluating query optimization. Compare the two answers below and choose the best one.

        Query: {original_query}
        
        Original Answer: {original_answer}
        
        Optimized Answer: {optimized_answer}
        
        Answer ONLY with exactly one word: "Optimized", "Original", or "Tie". Do not include any extra text.
        """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Post-process in case extra text is returned
    for option in ["Optimized", "Original", "Tie"]:
        if option.lower() in answer.lower():
            return option
    return answer  # Fallback, if no option is detected


def sanity_check(original_query, optimized_query):
    """Runs the LLM on both queries and compares the responses."""
    print(f"\n> Running sanity check for query optimization...")

    # Run LLM on both queries
    orig_answer = run_llm_query(original_query)
    opt_answer = run_llm_query(optimized_query)

    # Compare answers using embeddings
    embedding_judgment = compare_answers_with_embeddings(original_query, orig_answer, opt_answer)

    # Compare answers using an LLM judge
    llm_judgment = judge_with_llm(original_query, orig_answer, opt_answer)

    # Print results
    print("\n🔍 **Sanity Check Results**:")
    print(f"📝 Original Query: {original_query}")
    print(f"📝 Optimized Query: {optimized_query}")
    print(f"💬 Original Answer: {orig_answer}")
    print(f"💬 Optimized Answer: {opt_answer}")
    print(f"📊 LLM Judgment: {llm_judgment}")
    print(f"📊 Embedding Judgment: {embedding_judgment}")
    print(
        f"⚖ Final Decision: {'Optimized' if llm_judgment == 'Optimized' or embedding_judgment == 'Optimized' else 'Original'}")

    return {
        "\n\noriginal_query": original_query,
        "\n\noptimized_query": optimized_query,
        "\n\noriginal_answer": orig_answer,
        "\n\noptimized_answer": opt_answer,
        "\n\nllm_judgment": llm_judgment,
        "\n\nembedding_judgment": embedding_judgment,
        "\n\nfinal_decision": "Optimized" if llm_judgment == "Optimized" or embedding_judgment else "Original"
    }


if __name__ == "__main__":
    original_query = "What is the largest country in the world?"
    optimized_query = "which country has the largest surface on earth"

    result = sanity_check(original_query, optimized_query)
