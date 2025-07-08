# modules/query.py - Refactored for better organization
import numpy as np
from typing import Union, Optional, Dict, Callable, List, Tuple, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, PreTrainedModel, PreTrainedTokenizer

from configurations.config import MAX_RETRIES, QUALITY_THRESHOLD, MAX_NEW_TOKENS
from utility.embedding_utils import get_query_vector
from utility.logger import logger
from utility.similarity_calculator import calculate_similarities, SimilarityMethod

# Import performance monitoring and caching
try:
    from utility.performance import monitor_performance, track_performance
    from utility.cache import cache_query_result

    PERFORMANCE_AVAILABLE = True
    CACHE_AVAILABLE = True
except ImportError:
    logger.warning("Performance monitoring or caching not available")
    PERFORMANCE_AVAILABLE = False
    CACHE_AVAILABLE = False


    def monitor_performance(name):
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield

        return dummy_context()


    def track_performance(name=None):
        def decorator(func):
            return func

        return decorator


    def cache_query_result(model_name):
        def decorator(func):
            return func

        return decorator


# =====================================================
# EMBEDDING AND TEXT PROCESSING UTILITIES
# =====================================================

def extract_node_text_for_embedding(node) -> str:
    """
    Extract text from a LlamaIndex node exactly as LlamaIndex does for embedding.

    Args:
        node: LlamaIndex node object

    Returns:
        str: Text formatted for embedding, matching LlamaIndex's internal process
    """
    try:
        return node.get_content(metadata_mode=MetadataMode.EMBED)
    except Exception as e:
        logger.warning(f"Failed to extract embedding text: {e}. Using fallback.")
        return getattr(node, 'text', str(node))


def generate_embedding_with_normalization(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Generate embedding for text using LlamaIndex's method with optional normalization.

    Args:
        text: Text to embed
        embed_model: HuggingFace embedding model

    Returns:
        np.ndarray: Normalized embedding vector
    """
    with monitor_performance("embedding_generation"):
        logger.debug(f"Generating embedding for text: {text[:50]}...")

        embedding = embed_model.get_text_embedding(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        # Apply normalization if needed
        norm = np.linalg.norm(embedding_array)
        if norm > 1.1 or norm < 0.9:  # Not normalized
            embedding_array = embedding_array / norm
            logger.debug("Applied normalization to embedding")

        return embedding_array


def process_retrieved_nodes(nodes_with_scores) -> Tuple[List, List[float]]:
    """
    Extract nodes and scores from LlamaIndex retrieval results.

    Args:
        nodes_with_scores: LlamaIndex retrieval results

    Returns:
        Tuple of (nodes_list, scores_list)
    """
    nodes = [item.node for item in nodes_with_scores]
    scores = [item.score for item in nodes_with_scores]
    return nodes, scores


def batch_generate_embeddings(texts: List[str], embed_model: HuggingFaceEmbedding) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of texts to embed
        embed_model: HuggingFace embedding model

    Returns:
        np.ndarray: Array of embeddings (n_texts, embedding_dim)
    """
    with monitor_performance("batch_embedding_generation"):
        embeddings = [
            generate_embedding_with_normalization(text, embed_model)
            for text in texts
        ]
        return np.array(embeddings)


# =====================================================
# SIMILARITY AND FILTERING UTILITIES
# =====================================================

def calculate_similarity_scores(
        query_vector: np.ndarray,
        document_embeddings: np.ndarray,
        method: Union[SimilarityMethod, str, Callable],
        reference_scores: Optional[List[float]] = None
) -> np.ndarray:
    """
    Calculate similarity scores and optionally compare with reference scores.

    Args:
        query_vector: Query embedding vector
        document_embeddings: Document embedding matrix
        method: Similarity calculation method
        reference_scores: Optional reference scores for comparison

    Returns:
        np.ndarray: Calculated similarity scores
    """
    with monitor_performance("similarity_calculation"):
        similarities = calculate_similarities(query_vector, document_embeddings, method)

        # Log comparison if using cosine similarity and reference scores available
        if method == SimilarityMethod.COSINE and reference_scores is not None:
            _log_similarity_comparison(similarities, reference_scores)

        return similarities


def _log_similarity_comparison(manual_scores: np.ndarray, reference_scores: List[float]):
    """Log comparison between manual and reference similarity scores."""
    logger.info("Similarity score comparison:")
    for i, (manual, reference) in enumerate(zip(manual_scores, reference_scores)):
        diff = abs(manual - reference)
        logger.info(f"Doc {i}: Manual={manual:.6f}, Reference={reference:.6f}, Diff={diff:.6f}")


def filter_and_rank_results(
        similarity_scores: np.ndarray,
        node_contents: List[str],
        similarity_threshold: float,
        max_results: int
) -> List[str]:
    """
    Filter results by similarity threshold and return top-k ranked by score.

    Args:
        similarity_scores: Array of similarity scores
        node_contents: List of node content strings
        similarity_threshold: Minimum similarity score to include
        max_results: Maximum number of results to return

    Returns:
        List[str]: Filtered and ranked content strings
    """
    with monitor_performance("result_filtering_and_ranking"):
        # Vectorized filtering
        valid_mask = similarity_scores >= similarity_threshold

        if not np.any(valid_mask):
            logger.warning(f"No results above similarity threshold {similarity_threshold}")
            return []

        # Extract valid results
        valid_scores = similarity_scores[valid_mask]
        valid_contents = [content for i, content in enumerate(node_contents) if valid_mask[i]]

        # Sort by score (descending) and take top-k
        sorted_indices = np.argsort(valid_scores)[::-1][:max_results]
        ranked_contents = [valid_contents[i] for i in sorted_indices]

        logger.info(f"Filtered to {len(ranked_contents)} results from {len(node_contents)} candidates")
        return ranked_contents


# =====================================================
# CORE RETRIEVAL FUNCTIONS
# =====================================================

@track_performance("context_retrieval")
def retrieve_context_with_similarity(
        query: Union[str, np.ndarray],
        vector_db: VectorStoreIndex,
        embed_model: HuggingFaceEmbedding,
        max_results: int = 5,
        similarity_threshold: float = 0.5,
        similarity_method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE,
) -> str:
    """
    Retrieve relevant context using configurable similarity methods.

    Args:
        query: Query string or vector
        vector_db: LlamaIndex vector store
        embed_model: HuggingFace embedding model
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score threshold
        similarity_method: Method for calculating similarity

    Returns:
        str: Concatenated context from relevant documents
    """
    logger.info(f"Retrieving context using {similarity_method} similarity")

    # Input validation
    if not isinstance(query, (str, np.ndarray)):
        raise TypeError("Query must be a string or numpy array")

    # Step 1: Initial retrieval using LlamaIndex
    nodes_with_scores = _perform_initial_retrieval(query, vector_db, max_results)
    if not nodes_with_scores:
        return ''

    # Step 2: Process query and document embeddings
    query_vector = _prepare_query_vector(query, embed_model)
    nodes, reference_scores = process_retrieved_nodes(nodes_with_scores)

    # Step 3: Generate document embeddings
    document_embeddings = _generate_document_embeddings(nodes, embed_model)

    # Step 4: Calculate similarities using specified method
    similarity_scores = calculate_similarity_scores(
        query_vector, document_embeddings, similarity_method, reference_scores
    )

    # Step 5: Filter, rank, and format results
    node_contents = [node.get_content() for node in nodes]
    filtered_contents = filter_and_rank_results(
        similarity_scores, node_contents, similarity_threshold, max_results
    )

    logger.info(f"Retrieved {len(filtered_contents)} relevant documents")
    return "\n\n".join(filtered_contents)  # Double newline for better separation


def _perform_initial_retrieval(query, vector_db, max_results):
    """Perform initial retrieval using LlamaIndex."""
    with monitor_performance("llamaindex_retrieval"):
        retriever = vector_db.as_retriever(similarity_top_k=max_results)
        nodes_with_scores = retriever.retrieve(query)

        if nodes_with_scores:
            logger.info(f"LlamaIndex retrieved {len(nodes_with_scores)} initial candidates")
        else:
            logger.warning("No documents retrieved by LlamaIndex")

        return nodes_with_scores


def _prepare_query_vector(query, embed_model):
    """Prepare query vector from string or array input."""
    if isinstance(query, str):
        if embed_model is None:
            raise ValueError("Embedding model required for string queries")
        with monitor_performance("query_embedding"):
            return get_query_vector(query, embed_model)
    return query


def _generate_document_embeddings(nodes, embed_model):
    """Generate embeddings for document nodes."""
    with monitor_performance("document_embedding_extraction"):
        texts = [extract_node_text_for_embedding(node) for node in nodes]
        return batch_generate_embeddings(texts, embed_model)


# =====================================================
# TEXT GENERATION UTILITIES
# =====================================================

def prepare_generation_inputs(prompt: str, tokenizer: GPT2TokenizerFast, device: torch.device, max_length: int = 900):
    """
    Prepare tokenized inputs for text generation.

    Args:
        prompt: Input prompt text
        tokenizer: GPT2 tokenizer
        device: Target device
        max_length: Maximum sequence length

    Returns:
        dict: Tokenized inputs ready for generation
    """
    with monitor_performance("tokenization"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def generate_text_by_device(
        model: AutoModelForCausalLM,
        inputs: dict,
        device: torch.device,
        tokenizer: GPT2TokenizerFast
) -> torch.Tensor:
    """
    Generate text with device-specific optimizations.

    Args:
        model: Language model
        inputs: Tokenized inputs
        device: Target device
        tokenizer: Tokenizer for special tokens

    Returns:
        torch.Tensor: Generated token sequences
    """
    with monitor_performance("text_generation"):
        with torch.no_grad():
            base_params = {
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": tokenizer.eos_token_id
            }

            if device.type == "mps":
                # MPS-specific optimizations
                return model.generate(
                    **inputs,
                    max_new_tokens=min(MAX_NEW_TOKENS, 50),
                    use_cache=True,
                    **base_params
                )
            elif device.type == "cuda":
                # CUDA-specific optimizations
                return model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    top_p=0.9,
                    use_cache=True,
                    attention_mask=inputs.get('attention_mask'),
                    **base_params
                )
            else:
                # CPU generation
                return model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    **base_params
                )


def handle_gpu_memory_error(
        error: RuntimeError,
        model: AutoModelForCausalLM,
        inputs: dict,
        device: torch.device,
        tokenizer: GPT2TokenizerFast
) -> Tuple[str, str]:
    """
    Handle GPU memory errors with CPU fallback.

    Returns:
        Tuple of (generated_answer, device_used)
    """
    logger.warning(f"GPU memory issue: {error}")

    # Clear GPU cache
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # CPU fallback
    logger.info("Falling back to CPU generation")
    model_cpu = model.cpu()
    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_cpu.generate(
            **inputs_cpu,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Move model back to original device
    model.to(device)

    return answer, "cpu_fallback"


# =====================================================
# MAIN QUERY PROCESSING
# =====================================================

@track_performance("answer_quality_evaluation")
def evaluate_answer_quality(
        answer: str,
        question: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> float:
    """
    Evaluate the quality of a generated answer.

    Currently returns a placeholder score.
    TODO: Implement actual quality evaluation logic.
    """
    logger.info("Evaluating answer quality")
    # Placeholder - implement actual evaluation logic
    score = 1.0
    logger.info(f"Quality score: {score}")
    return score


@cache_query_result("distilgpt2")
@track_performance("complete_query_processing")
def process_query_with_context(
        prompt: str,
        model: Union[PreTrainedModel, Any],
        tokenizer: Union[PreTrainedTokenizer, Any],
        device: torch.device,
        vector_db: Optional[VectorStoreIndex] = None,
        embedding_model: Optional[HuggingFaceEmbedding] = None,
        max_retries: int = MAX_RETRIES,
        quality_threshold: float = QUALITY_THRESHOLD,
        similarity_method: Union[SimilarityMethod, str, Callable] = SimilarityMethod.COSINE
) -> Dict[str, Union[str, float, int, None]]:
    """
    Process a query with optional context retrieval and iterative improvement.

    This is the main entry point for query processing with RAG capabilities.
    """
    logger.info(f"Processing query with {similarity_method} similarity")

    # Ensure model is on correct device
    _ensure_model_on_device(model, device)

    try:
        for attempt in range(max_retries + 1):
            try:
                # Generate answer for current attempt
                result = _generate_single_answer(
                    prompt, model, tokenizer, device, vector_db, embedding_model, similarity_method)

                # Check if answer meets quality threshold
                if result["score"] >= quality_threshold or attempt == max_retries:
                    result.update({
                        "attempts": attempt + 1,
                        "similarity_method": str(similarity_method)
                    })
                    logger.info("Query processing completed successfully")
                    return result

                # Improve prompt for next iteration
                prompt = _improve_prompt(prompt, result["answer"], model, tokenizer, device)

            except RuntimeError as err:
                if _is_gpu_memory_error(err):
                    answer, device_used = handle_gpu_memory_error(err, model, {}, device, tokenizer)
                    return _create_result_dict(prompt, answer, 1.0, attempt + 1,
                                               f"GPU fallback: {err}", device_used, similarity_method)
                else:
                    raise err

        # If we reach here, max retries exceeded
        return _create_result_dict(prompt, "No satisfactory answer generated", 0.0,
                                   max_retries + 1, None, str(device), similarity_method)

    except Exception as err:
        logger.error(f"Error during query processing: {err}")
        return _create_result_dict(prompt, f"Error: {err}", 0.0, 0, str(err), str(device), similarity_method)


def _ensure_model_on_device(model: AutoModelForCausalLM, device: torch.device):
    """Ensure model is on the correct device."""
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.info(f"Moving model from {model_device} to {device}")
        model.to(device)


def _generate_single_answer(prompt, model, tokenizer, device, vector_db, embedding_model, similarity_method):
    """Generate a single answer attempt."""
    # Context retrieval
    augmented_prompt = _prepare_prompt_with_context(
        prompt, vector_db, embedding_model, similarity_method
    )

    # Generate answer
    inputs = prepare_generation_inputs(augmented_prompt, tokenizer, device)
    outputs = generate_text_by_device(model, inputs, device, tokenizer)

    # Process output
    with monitor_performance("answer_processing"):
        outputs = outputs.cpu()
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = evaluate_answer_quality(answer, prompt, model, tokenizer, device)

    return {"answer": answer, "score": score}


def _prepare_prompt_with_context(prompt, vector_db, embedding_model, similarity_method):
    """Prepare prompt with retrieved context if available."""
    if vector_db is not None:
        with monitor_performance("context_retrieval_and_prompt_construction"):
            context = retrieve_context_with_similarity(
                prompt, vector_db, embedding_model, similarity_method=similarity_method
            )
            return f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"

    return prompt


def _improve_prompt(original_prompt, previous_answer, model, tokenizer, device):
    """Improve prompt based on previous answer."""
    return rephrase_query(original_prompt, previous_answer, model, tokenizer, device)


def _is_gpu_memory_error(error):
    """Check if error is related to GPU memory."""
    error_str = str(error).lower()
    return "mps" in error_str or "out of memory" in error_str


def _create_result_dict(question, answer, score, attempts, error, device_used, similarity_method):
    """Create standardized result dictionary."""
    return {
        "question": question,
        "answer": answer,
        "score": score,
        "attempts": attempts,
        "error": error,
        "device_used": device_used,
        "similarity_method": str(similarity_method)
    }


@track_performance("query_rephrasing")
def rephrase_query(
        original_prompt: str,
        previous_answer: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> str:
    """
    Rephrase query based on previous answer to improve results.
    """
    logger.info("Rephrasing query for improvement")

    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"Previous answer: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )

    inputs = prepare_generation_inputs(rephrase_prompt, tokenizer, device, max_length=800)

    with monitor_performance("rephrase_generation"):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

    outputs = outputs.cpu()
    rephrased_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    logger.info("Query rephrased successfully")
    return rephrased_query
