# modules/query.py
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
from utility.logger import logger

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


    # Create dummy decorators
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
    with monitor_performance("vector_similarity_calculation"):
        logger.info("Calculating cosine similarity between two vectors.")
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0.0 or norm2 == 0.0:
            logger.warning(
                f"One or both vectors are zero vectors, returning similarity as 0.0. vector1: {vector1} vector2: {vector2}")
            return 0.0
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
    with monitor_performance("llamaindex_embedding_retrieval"):
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


@track_performance("context_retrieval")
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
    with monitor_performance("llamaindex_retrieval"):
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
        with monitor_performance("query_vector_generation"):
            query_vector = get_query_vector(query, embed_model)
    elif isinstance(query, np.ndarray):
        query_vector = query

    # Extract nodes and their LlamaIndex-compatible text
    with monitor_performance("text_extraction_and_embedding"):
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
        with monitor_performance("manual_similarity_calculation"):
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
    with monitor_performance("content_extraction_and_filtering"):
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
    Use retrieve_context_aligned_to_llama_index for better LlamaIndex compatibility.
    """
    logger.warning(
        "Using original retrieve_context. Consider using retrieve_context_aligned_to_llama_index for better compatibility.")
    return retrieve_context_aligned_to_llama_index(query, vector_db, embed_model, top_k, similarity_cutoff)


@track_performance("answer_quality_evaluation")
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


@cache_query_result("distilgpt2")
@track_performance("complete_query_processing")
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
    GPU-optimized queries the language model with the given prompt and retrieves an answer.
    Now includes performance monitoring and caching.

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
    logger.info("Starting GPU-optimized query process.")

    # Ensure model is on the correct device
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.info(f"Moving model from {model_device} to {device}")
        model = model.to(device)

    try:
        answer: str = ""
        score: float = 0.0
        attempt: int = 0
        current_prompt: str = prompt

        while attempt <= max_retries:
            try:
                # Context retrieval (runs on CPU - that's fine)
                if vector_db is not None:
                    if use_improved_retrieval:
                        retrieved_context = retrieve_context_aligned_to_llama_index(
                            current_prompt, vector_db, embedding_model
                        )
                    else:
                        retrieved_context = retrieve_context(
                            current_prompt, vector_db, embedding_model
                        )

                    with monitor_performance("prompt_construction"):
                        augmented_prompt = f"Context: {retrieved_context}\n\nQuestion: {current_prompt}\nAnswer:"
                else:
                    augmented_prompt = current_prompt

                # GPU-optimized tokenization
                with monitor_performance("tokenization"):
                    inputs = tokenizer(
                        augmented_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=900,  # Conservative for MPS stability
                        padding=True
                    )

                    # Move inputs to device efficiently
                    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                # GPU generation with device-specific optimizations
                with monitor_performance("text_generation"):
                    with torch.no_grad():  # Save GPU memory
                        if device.type == "mps":
                            # MPS-optimized generation
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=min(MAX_NEW_TOKENS, 50),  # Conservative for MPS
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                            )
                        elif device.type == "cuda":
                            # CUDA-optimized generation
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                                attention_mask=inputs.get('attention_mask')
                            )
                        else:
                            # CPU generation
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id
                            )

                # Move output back to CPU for decoding (more efficient)
                with monitor_performance("answer_processing"):
                    outputs = outputs.cpu()
                    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    score = evaluate_answer_quality(answer, current_prompt, model, tokenizer, device)

                if score >= quality_threshold or attempt == max_retries:
                    logger.info("GPU-optimized query completed successfully.")
                    return {
                        "question": prompt,
                        "answer": answer,
                        "score": score,
                        "attempts": attempt + 1,
                        "error": None,
                        "device_used": str(device)
                    }

                current_prompt = rephrase_query(current_prompt, answer, model, tokenizer, device)
                attempt += 1

            except RuntimeError as err:
                # Handle GPU memory issues gracefully
                if "MPS" in str(err) or "out of memory" in str(err).lower():
                    logger.warning(f"GPU memory issue: {err}")

                    # Clear GPU cache
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()

                    # Fallback to CPU for this generation
                    logger.info("Falling back to CPU for this generation")
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

                    # Move model back to GPU
                    model = model_cpu.to(device)

                    return {
                        "question": prompt,
                        "answer": answer,
                        "score": 1.0,
                        "attempts": attempt + 1,
                        "error": f"GPU fallback: {str(err)}",
                        "device_used": "cpu_fallback"
                    }
                else:
                    raise err

            except Exception as err:
                logger.error(f"Error during GPU query process: {err}")
                return {
                    "question": prompt,
                    "answer": f"Error: {str(err)}",
                    "score": 0.0,
                    "attempts": attempt + 1,
                    "error": str(err)
                }

        return {
            "question": prompt,
            "answer": answer or "No answer generated",
            "score": score,
            "attempts": attempt + 1,
            "error": None,
            "device_used": str(device)
        }

    except Exception as err:
        logger.error(f"Error during GPU query process: {err}")
        return {
            "question": prompt,
            "answer": "Error during generation.",
            "score": 0.0,
            "attempts": 0,
            "error": str(err)
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
    GPU-optimized query rephrasing based on the previous answer to improve the response.

    Args:
        original_prompt (str): Original query prompt.
        previous_answer (str): Previous answer generated by the model.
        model (AutoModelForCausalLM): Language model used for rephrasing.
        tokenizer (GPT2TokenizerFast): Tokenizer for processing text.
        device (torch.device): Device to run the rephrasing on.

    Returns:
        str: Rephrased query.
    """
    logger.info("GPU-optimized query rephrasing.")
    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"The previous answer was: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )

    with monitor_performance("rephrase_tokenization"):
        inputs = tokenizer(
            rephrase_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=800
        ).to(device)

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
    logger.info("Query rephrased successfully.")
    return rephrased_query