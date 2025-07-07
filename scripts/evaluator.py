# evaluator.py
from typing import Tuple, Union
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from configurations.config import HF_MODEL_NAME, LLM_MODEL_NAME
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
