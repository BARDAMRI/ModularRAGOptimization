# evaluator.py - Final polished version
from typing import Tuple, Union, List, Dict, Any, Optional
import torch
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM
)
from configurations.config import HF_MODEL_NAME, LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE
from utility.logger import logger
from utility.similarity_calculator import calculate_similarity, calculate_similarities, SimilarityMethod
from utility.embedding_utils import get_text_embedding


def load_llm() -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Load the LLM model and tokenizer.
    Returns (model, tokenizer) to match process_query_with_context expectations.
    """
    logger.info("Loading LLM model and tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
        logger.info(f"Loaded causal language model: {LLM_MODEL_NAME}")
    except ValueError as e:
        logger.warning(f"Failed to load as causal model: {e}")
        model = AutoModel.from_pretrained(LLM_MODEL_NAME)
        logger.warning(f"Loaded masked language model instead: {LLM_MODEL_NAME}")

    return model, tokenizer


def load_model() -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Alias for load_llm() to match main.py expectations."""
    return load_llm()


def load_embedding_model() -> HuggingFaceEmbedding:
    """Load using LlamaIndex's HuggingFaceEmbedding wrapper."""
    logger.info(f"Loading embedding model: {HF_MODEL_NAME}")
    embedding_model = HuggingFaceEmbedding(model_name=HF_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
    return embedding_model


def run_llm_query(
        query: str,
        model: Union[PreTrainedModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer
) -> str:
    """Run a query using the LLM."""
    logger.info(f"Running query: {query[:100]}...")  # Truncate long queries in logs
    inputs = tokenizer(query, return_tensors="pt")

    if hasattr(model, "generate"):
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated result: {result[:100]}...")  # Truncate long results
    else:
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        result = str(embeddings)
        logger.info("Computed embeddings from masked model")

    return result


def enumerate_top_documents(
        i: int,
        num: int,
        query: str,
        index: Any,
        embedding_model: HuggingFaceEmbedding,  # Fixed: Added type annotation
        top_k: int = 5,
        convert_to_vector: bool = False,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> Dict[str, Any]:
    """
    Enumerate top documents using the new similarity calculator.

    Args:
        i: Current query vector_db
        num: Total number of queries
        query: Query string
        index: Vector database vector_db
        embedding_model: HuggingFace embedding model
        top_k: Number of top documents to retrieve
        convert_to_vector: Whether to convert query to vector
        similarity_method: Similarity calculation method

    Returns:
        Dict containing query results and top documents
    """
    logger.info(f"Enumerating top documents for query #{i + 1} of {num}: {query[:50]}... using {similarity_method}")

    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    if convert_to_vector:
        query_vector = get_text_embedding(query, embedding_model)
    else:
        query_vector = query

    results = retriever.retrieve(query_vector)

    # Enhanced scoring using configurable similarity methods
    enhanced_results = []
    if convert_to_vector and results:
        # Get embeddings for all retrieved documents
        doc_texts = [node_with_score.node.get_content() for node_with_score in results]
        doc_embeddings = np.array([get_text_embedding(text, embedding_model) for text in doc_texts])

        # Calculate similarities using the new system
        if len(doc_embeddings) > 0:
            similarities = calculate_similarities(query_vector, doc_embeddings, similarity_method)

            # Combine with original results - Fixed: Use different variable name to avoid confusion
            for rank, (node_with_score, custom_score) in enumerate(zip(results, similarities), 1):
                content = node_with_score.node.get_content()
                enhanced_results.append({
                    "rank": rank,
                    "original_score": getattr(node_with_score, "score", None),
                    "custom_score": float(custom_score),
                    "similarity_method": str(similarity_method),
                    "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
                })
    else:
        # Fallback to original scoring
        for rank, node_with_score in enumerate(results, start=1):
            content = node_with_score.node.get_content()
            enhanced_results.append({
                "rank": rank,
                "original_score": getattr(node_with_score, "score", None),
                "custom_score": None,
                "similarity_method": "llamaindex_default",
                "content": content[:500] + ("...[truncated]" if len(content) > 500 else "")
            })

    result = {
        "query": query,
        "similarity_method": str(similarity_method),
        "convert_to_vector": convert_to_vector,
        "top_documents": enhanced_results,
        "total_documents": len(enhanced_results)
    }
    logger.info(f"Top documents enumerated using {similarity_method}: {len(enhanced_results)} documents")
    return result


def hill_climb_documents(
        i: int,
        num: int,
        query: str,
        index: Any,
        llm_model: Union[PreTrainedModel, AutoModelForCausalLM],
        tokenizer: PreTrainedTokenizer,
        embedding_model: HuggingFaceEmbedding,  # Fixed: Added type annotation
        top_k: int = 5,
        max_tokens: int = 100,
        temperature: float = 0.07,
        convert_to_vector: bool = False,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> Dict[str, Any]:
    """
    Perform hill climbing with configurable similarity methods.

    Args:
        i: Current query vector_db
        num: Total number of queries
        query: Query string
        index: Vector database vector_db
        llm_model: Language model for generating answers
        tokenizer: Tokenizer for the language model
        embedding_model: HuggingFace embedding model
        top_k: Number of top documents to retrieve
        max_tokens: Maximum tokens for generated answers
        convert_to_vector: Whether to convert query to vector
        similarity_method: Similarity calculation method

    Returns:
        Dict containing query results and best answer

    Parameters
    ----------
    similarity_method
    convert_to_vector
    max_tokens
    top_k
    embedding_model
    tokenizer
    llm_model
    index
    i
    query
    num
    temperature
    """
    logger.info(f"Hill climbing for query #{i + 1} of {num}: {query[:50]}... using {similarity_method}")

    retriever = index.as_retriever()
    retriever.retrieve_mode = "embedding"
    retriever.similarity_top_k = top_k

    if convert_to_vector:
        query_vector = get_text_embedding(query, embedding_model)
    else:
        query_vector = query

    results = retriever.retrieve(query_vector)

    best_score = -1.0
    best_answer = None
    best_context = None
    best_method_info = {"method": str(similarity_method), "score": best_score}

    contexts = [node_with_score.node.get_content() for node_with_score in results]
    if not contexts:
        logger.warning("No contexts retrieved.")
        return {"query": query, "answer": None, "context": None, "method_info": best_method_info}

    # Generate answers for all contexts
    prompts = [f"Context:\n{context}\n\nQuestion: {query}\nAnswer:" for context in contexts]

    # Handle tokenization with proper truncation
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = llm_model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=False)
        answers = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        return {"query": query, "answer": None, "context": None, "method_info": best_method_info}

    # Use modular similarity calculator for scoring
    if convert_to_vector:
        query_emb = get_text_embedding(query, embedding_model)

        for context, answer in zip(contexts, answers):
            if not answer.strip():
                continue

            try:
                answer_emb = get_text_embedding(answer, embedding_model)
                # Use the new similarity calculator
                score = calculate_similarity(query_emb, answer_emb, similarity_method)

                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_context = context
                    best_method_info = {
                        "method": str(similarity_method),
                        "score": float(best_score)
                    }
            except Exception as e:
                logger.warning(f"Error calculating similarity for answer: {e}")
                continue
    else:
        # Fallback for non-vector mode
        for context, answer in zip(contexts, answers):
            if not answer.strip():
                continue

            # Simple word overlap heuristic
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            score = len(query_words & answer_words) / max(len(query_words), 1)

            if score > best_score:
                best_score = score
                best_answer = answer
                best_context = context
                best_method_info = {
                    "method": "word_overlap",
                    "score": float(best_score)
                }

    logger.info(f"Best answer selected using {similarity_method} with score {best_score:.4f}")
    return {
        "query": query,
        "answer": best_answer,
        "context": best_context,
        "method_info": best_method_info,
        "total_contexts_evaluated": len(contexts)
    }


def compare_answers_with_embeddings(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        similarity_method: Union[SimilarityMethod, str] = SimilarityMethod.COSINE
) -> str:
    """
    Compare answers using the new similarity calculator system.

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        similarity_method: Similarity calculation method

    Returns:
        str: "Optimized", "Original", or "Tie"
    """
    logger.info(f"Comparing answers using {similarity_method} similarity")

    try:
        embedding_model = load_embedding_model()

        # Generate embeddings
        query_embedding = get_text_embedding(user_query, embedding_model)
        orig_embedding = get_text_embedding(original_answer, embedding_model)
        opt_embedding = get_text_embedding(optimized_answer, embedding_model)

        # Use modular similarity calculator
        sim_orig = calculate_similarity(query_embedding, orig_embedding, similarity_method)
        sim_opt = calculate_similarity(query_embedding, opt_embedding, similarity_method)

        logger.info(f"Similarity scores using {similarity_method} - Original: {sim_orig:.4f}, Optimized: {sim_opt:.4f}")

        # More robust tie detection
        score_diff = abs(sim_opt - sim_orig)
        if score_diff < 0.001:  # Very close scores
            return "Tie"
        elif score_diff < 0.01:  # Close scores - use relative difference
            relative_diff = score_diff / max(abs(sim_orig), abs(sim_opt), 1e-6)
            if relative_diff < 0.05:  # Less than 5% relative difference
                return "Tie"

        return "Optimized" if sim_opt > sim_orig else "Original"

    except Exception as e:
        logger.error(f"Error in answer comparison: {e}")
        return "Tie"


def multi_method_comparison(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        methods: Optional[List[Union[SimilarityMethod, str]]] = None
) -> Dict[str, Any]:
    """
    Compare answers using multiple similarity methods for robust evaluation.

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        methods: List of similarity methods to use

    Returns:
        Dict containing results from all methods with final consensus
    """
    if methods is None:
        methods = [SimilarityMethod.COSINE, SimilarityMethod.DOT_PRODUCT, SimilarityMethod.EUCLIDEAN]

    logger.info(f"Multi-method comparison using {len(methods)} similarity methods")

    results = {}
    votes = {"Original": 0, "Optimized": 0, "Tie": 0}
    scores = {"Original": [], "Optimized": [], "Tie": []}

    for method in methods:
        try:
            result = compare_answers_with_embeddings(user_query, original_answer, optimized_answer, method)
            results[str(method)] = result
            votes[result] += 1

            # Could add actual similarity scores here for weighted voting
            # scores[result].append(score)

        except Exception as e:
            logger.warning(f"Error with method {method}: {e}")
            results[str(method)] = "Tie"
            votes["Tie"] += 1

    # Determine consensus
    consensus = max(votes, key=votes.get)
    confidence = votes[consensus] / len(methods)

    final_result = {
        "individual_results": results,
        "votes": votes,
        "consensus": consensus,
        "confidence": confidence,
        "methods_used": [str(m) for m in methods],
        "total_methods": len(methods)
    }

    logger.info(f"Multi-method consensus: {consensus} (confidence: {confidence:.2f})")
    return final_result


def judge_with_llm(
        user_query: str,
        original_answer: str,
        optimized_answer: str,
        model: Optional[Union[PreTrainedModel, AutoModelForCausalLM]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None
) -> str:
    """
    Judge answers using a language model (LLM).

    Args:
        user_query: The original query
        original_answer: First answer to compare
        optimized_answer: Second answer to compare
        model: LLM model instance (optional)
        tokenizer: Tokenizer for the LLM (optional)
        device: Device to run the model on (optional)

    Returns:
        str: "Optimized", "Original", or "Tie"
    """
    logger.info("Judging answers with LLM.")

    if model is None or tokenizer is None:
        model, tokenizer = load_llm()

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if hasattr(model, 'device') and str(model.device) != device:
            model = model.to(device)

        prompt = f"""You are an AI judge evaluating query optimization. Compare the two answers below and choose the best one.

Query: {user_query}

Original Answer: {original_answer}

Optimized Answer: {optimized_answer}

Answer ONLY with exactly one word: "Optimized", "Original", or "Tie". Do not include any extra text.
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        if device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        logger.info(f"LLM judgment result: {answer}")

        # More robust answer parsing
        answer_lower = answer.lower()
        for option in ["optimized", "original", "tie"]:
            if option in answer_lower:
                return option.capitalize()

        # Fallback - look for partial matches
        if "optim" in answer_lower:
            return "Optimized"
        elif "origin" in answer_lower:
            return "Original"
        else:
            return "Tie"

    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return "Tie"


def advanced_sanity_check(
        user_query: str,
        optimized_user_query: str,
        vector_db: Any,
        vector_query: Optional[Any] = None,
        similarity_methods: Optional[List[Union[SimilarityMethod, str]]] = None
) -> Dict[str, Any]:
    """
    Enhanced sanity check using multiple similarity methods.

    Provides more robust evaluation by testing different similarity approaches.

    Args:
        user_query: Original user query
        optimized_user_query: Optimized version of the query
        vector_db: Vector database instance
        vector_query: Optional vector representation of the query
        similarity_methods: List of similarity methods to use

    Returns:
        Dict containing comprehensive evaluation results
    """
    if similarity_methods is None:
        similarity_methods = [SimilarityMethod.COSINE, SimilarityMethod.DOT_PRODUCT]

    print(f"\n> Running advanced sanity check with {len(similarity_methods)} similarity methods...")

    try:
        model, tokenizer = load_llm()

        # Get answers (simplified for this example)
        orig_answer = run_llm_query(user_query, model, tokenizer)
        opt_answer = run_llm_query(optimized_user_query, model, tokenizer)

        # Multi-method embedding comparison
        multi_method_result = multi_method_comparison(user_query, orig_answer, opt_answer, similarity_methods)

        # Traditional LLM judgment
        llm_judgment = judge_with_llm(user_query, orig_answer, opt_answer, model=model, tokenizer=tokenizer)

        # Display results
        print("\n🔍 **Advanced Sanity Check Results**:")
        print(f"📝 Original Query: {user_query}")
        print(f"📝 Optimized Query: {optimized_user_query}")
        print(f"💬 Original Answer: {orig_answer[:200]}...")
        print(f"💬 Optimized Answer: {opt_answer[:200]}...")
        print(f"📊 LLM Judgment: {llm_judgment}")
        print(
            f"📊 Multi-method Results: {multi_method_result['consensus']} (confidence: {multi_method_result['confidence']:.2f})")
        print(f"📊 Individual Method Results: {multi_method_result['individual_results']}")

        # Determine final decision with weighted voting
        embedding_confidence = multi_method_result['confidence']
        llm_confidence = 0.7  # Fixed weight for LLM judgment

        final_decision = multi_method_result['consensus']
        if llm_judgment != multi_method_result['consensus'] and llm_confidence > embedding_confidence:
            final_decision = llm_judgment
            decision_reason = "LLM judgment overrode multi-method consensus due to higher confidence"
        else:
            decision_reason = "Multi-method consensus accepted"

        print(f"⚖️ Final Decision: {final_decision} ({decision_reason})")

        return {
            "original_query": user_query,
            "optimized_query": optimized_user_query,
            "original_answer": orig_answer,
            "optimized_answer": opt_answer,
            "llm_judgment": llm_judgment,
            "multi_method_result": multi_method_result,
            "final_decision": final_decision,
            "decision_reason": decision_reason,
            "confidence_scores": {
                "embedding_confidence": embedding_confidence,
                "llm_confidence": llm_confidence
            },
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in advanced sanity check: {e}")
        return {
            "error": str(e),
            "success": False
        }
