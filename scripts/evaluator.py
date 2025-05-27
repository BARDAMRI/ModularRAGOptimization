# evaluator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from config import HF_MODEL_NAME
from datetime import datetime
from matrics.results_logger import ResultsLogger, plot_score_distribution

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


def enumerate_top_documents(i, num, query, index, embedding_model, top_k=5):
    """
    Retrieves and displays top-k documents for the query using embedding similarity.
    Returns a summary dictionary and logs it to results.
    """
    print(f"\n🔍 Starting enumerating for query #{i + 1} of {num}: {query}")
    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    results = retriever.retrieve(query)
    logger = ResultsLogger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ["results", "enumerate"]
    if top_k:
        filename_parts.append(f"top{top_k}")
    filename_parts.append(timestamp)
    name = "_".join(filename_parts)
    plot_score_distribution(name)

    top_docs = []
    print(f"\n📌 Query: {query}\n")

    for i, node_with_score in enumerate(results):
        # Try to get the similarity score if present
        score = node_with_score.score if hasattr(node_with_score, "score") else None
        # Get the content from the node
        content = node_with_score.node.get_content()

        print(f"📄 Rank {i + 1}")
        print(f"🔢 Similarity Score: {score:.4f}" if score is not None else "🔢 Similarity Score: N/A")
        print(f"🧾 Content Preview:\n{content[:300]}{'...' if len(content) > 300 else ''}")
        print("-" * 50)

        # Prepare document summary for result dictionary
        top_docs.append({
            "rank": i + 1,
            "score": score,
            "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
        })

    result = {
        "query": query,
        "top_documents": top_docs
    }

    logger.log(result)
    return result


# Hill climbing over top-k retrieved documents to find best answer/context
def hill_climb_documents(i, num, query, index, llm_model, tokenizer, embedding_model, top_k=5, max_tokens=100):
    """
    Performs hill climbing over top-k retrieved documents to find the best context for generating an answer.
    Returns the best answer and its associated score and document.
    """
    print(f"\n🔍 Starting hill climbing for query #{i + 1} of {num}: {query}")
    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    results = retriever.retrieve(query)

    best_score = -1.0
    best_answer = None
    best_context = None

    for i, node_with_score in enumerate(results):
        context = node_with_score.node.get_content()
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = llm_model.generate(**inputs, max_new_tokens=max_tokens)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        query_emb = torch.tensor(embedding_model.get_text_embedding(query))
        answer_emb = torch.tensor(embedding_model.get_text_embedding(answer))
        score = util.pytorch_cos_sim(query_emb, answer_emb).item()

        print(f"\n🔎 Candidate #{i + 1}")
        print(f"📄 Similarity Score: {score:.4f}")
        print(f"💬 Answer:\n{answer}")
        print("-" * 50)

        if score > best_score:
            best_score = score
            best_answer = answer
            best_context = context

    print("\n✅ Best Answer Selected:")
    print(f"📊 Similarity Score: {best_score:.4f}")
    print(f"🧾 Context Preview:\n{best_context[:300]}{'...' if len(best_context) > 300 else ''}")
    print(f"💬 Answer: {best_answer}")

    return {
        "query": query,
        "answer": best_answer,
        "score": best_score,
        "context": best_context
    }


if __name__ == "__main__":
    original_query = "What is the largest country in the world?"
    optimized_query = "which country has the largest surface on earth"

    result = sanity_check(original_query, optimized_query)
