# modules/query.py - Refactored for better organization
import re
from typing import Union, Optional, Dict, Callable, List, Tuple, Any

import numpy as np
import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, PreTrainedModel, PreTrainedTokenizer

from configurations.config import INDEX_SOURCE_URL
from configurations.config import MAX_RETRIES, QUALITY_THRESHOLD, MAX_NEW_TOKENS, TEMPERATURE
from modules.model_loader import load_model
from utility.embedding_utils import get_query_vector
from utility.logger import logger
from utility.similarity_calculator import calculate_similarities, SimilarityMethod
from vector_db.indexer import load_vector_db

# =====================================================


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

def extract_node_text_for_embedding(node: Union[TextNode, List[str]]) -> Union[str, List[str]]:
    """
    Extract text from a LlamaIndex node exactly as LlamaIndex does for embedding.

    Args:
        node: LlamaIndex node object or  LlamaIndex node list

    Returns:
        str: Text formatted for embedding, matching LlamaIndex's internal process
    """
    try:
        if isinstance(node, list):
            if hasattr(node[0], 'text'):
                return [str(n.text) for n in node]
            else:
                return [n.get_content(metadata_mode=MetadataMode.EMBED) for n in node]
        else:
            if hasattr(node, 'text'):
                return str(node.text)
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
        nodes_with_scores = vector_db.retrieve(query)

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

# REPLACE YOUR EXISTING prepare_generation_inputs FUNCTION WITH THIS:
def prepare_generation_inputs(prompt: str, tokenizer, device: torch.device, max_length: int = 900):
    """
    MPS-safe input preparation.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # MPS-safe processing
    safe_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if device.type == "mps":
                if value.dtype == torch.bfloat16:
                    safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                elif key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    safe_inputs[key] = value.to(dtype=torch.long, device=device)
                else:
                    safe_inputs[key] = value.to(device)
            else:
                safe_inputs[key] = value.to(device)
        else:
            safe_inputs[key] = value

    return safe_inputs


def force_float32_on_mps_model(model):
    """Force all model parameters to float32 for MPS compatibility."""
    if next(model.parameters()).device.type == "mps":
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                print(f"Converting {name} from bfloat16 to float32")
                param.data = param.data.to(torch.float32)

        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.bfloat16:
                print(f"Converting buffer {name} from bfloat16 to float32")
                buffer.data = buffer.data.to(torch.float32)
    return model


def safe_mps_generate(model, tokenizer, inputs, device, max_tokens=64, temperature=0.7):
    """MPS-safe text generation that handles bfloat16 issues."""

    print(f"🔧 Called safe_mps_generate with device: {device}")  # Debug

    # Step 1: Force model to float32 if on MPS
    if device.type == "mps":
        print("🔧 Converting model to float32 for MPS")  # Debug
        model = force_float32_on_mps_model(model)

    # Step 2: Ensure all inputs are correct dtype
    safe_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if device.type == "mps":
                # For MPS, ensure proper dtypes
                if value.dtype == torch.bfloat16:
                    print(f"🔧 Converting input {key} from bfloat16 to float32")  # Debug
                    safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                elif key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    # These should be long tensors
                    safe_inputs[key] = value.to(dtype=torch.long, device=device)
                else:
                    # Default to float32 for MPS
                    if torch.is_floating_point(value):
                        safe_inputs[key] = value.to(dtype=torch.float32, device=device)
                    else:
                        safe_inputs[key] = value.to(device)
            else:
                # For non-MPS devices, just move to device
                safe_inputs[key] = value.to(device)
        else:
            safe_inputs[key] = value

    print("🔍 Input tensor dtypes:")
    for key, value in safe_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype} on {value.device}")

    # Step 3: Set model to eval mode and disable gradients
    model.eval()

    try:
        with torch.no_grad():
            print(f"🚀 Starting generation on {device}")  # Debug

            # Use conservative generation settings for MPS
            if device.type == "mps":
                outputs = model.generate(
                    **safe_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding is more stable
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            else:
                # Standard generation for other devices
                outputs = model.generate(
                    **safe_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

        print("✅ Generation completed successfully")
        return outputs

    except RuntimeError as e:
        print(f"❌ Generation failed: {e}")

        if "bfloat16" in str(e).lower() or "mps" in str(e).lower():
            print(f"🔄 MPS generation failed, falling back to CPU: {e}")

            # Move to CPU and try again
            model_cpu = model.cpu()
            inputs_cpu = {k: v.cpu() for k, v in safe_inputs.items()}

            with torch.no_grad():
                outputs = model_cpu.generate(
                    **inputs_cpu,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            # Move model back to original device
            try:
                model.to(device)
                print(f"✅ Model moved back to {device}")
            except Exception as move_error:
                print(f"⚠️  Warning: Could not move model back to {device}: {move_error}")

            return outputs
        else:
            raise e


def generate_text_by_device(model, inputs, device, tokenizer, max_tokens: int = 64, temperature: float = 0.7):
    """
    MPS-safe replacement for generate_text_by_device.
    """
    return safe_mps_generate(model, tokenizer, inputs, device, max_tokens, temperature=temperature)


def extract_answer_from_output(raw_output: str, original_prompt: str) -> str:
    """
    Extract clean answer from model output, handling various formats.
    """
    logger.info(f"🔍 Debug - Raw output: {repr(raw_output)}")
    logger.info(f"🔍 Debug - Original prompt: {repr(original_prompt)}")

    # Method 1: Try to find the first occurrence of "Answer:" and extract what follows
    if "Answer:" in raw_output:
        # Split by "Answer:" and take the content after the first occurrence
        parts = raw_output.split("Answer:")
        if len(parts) > 1:
            answer_part = parts[1]

            # Clean up the answer and Remove quotes if they wrap the entire answer
            answer_part = answer_part.strip()
            if answer_part.startswith('"') and '"' in answer_part[1:]:
                # Find the end quote
                end_quote_idx = answer_part.find('"', 1)
                if end_quote_idx != -1:
                    answer_part = answer_part[1:end_quote_idx]

            # Stop at the next "Question:" if it appears
            if "Question:" in answer_part:
                answer_part = answer_part.split("Question:")[0]

            # cleanup
            answer_part = answer_part.strip()

            # Remove trailing punctuation if it looks like start of new question
            if answer_part.endswith(('" Question', '" Q')):
                answer_part = answer_part.split('"')[0]

            logger.info(f"🔍 Debug - Extracted answer: {repr(answer_part)}")
            return answer_part.strip()

    # Method 2: If no "Answer:" found, try to extract from the end
    if original_prompt.strip() in raw_output:
        remaining = raw_output.replace(original_prompt.strip(), "", 1).strip()
        if remaining:
            # Take only the first sentence/phrase
            sentences = remaining.split('.')
            if sentences and sentences[0].strip():
                answer = sentences[0].strip()
                logger.info(f"🔍 Debug - Fallback answer: {repr(answer)}")
                return answer

    # Method 3: Last resort - try to find any reasonable answer

    quoted_matches = re.findall(r'"([^"]*)"', raw_output)
    if quoted_matches:
        # Return the first quoted string that's not the question
        for match in quoted_matches:
            if match.lower() not in original_prompt.lower() and len(match.strip()) > 0:
                logger.info(f"🔍 Debug - Quoted answer: {repr(match)}")
                return match.strip()

    # Final fallback
    logger.info(f"🔍 Debug - No clean answer found, returning cleaned raw output")
    clean_output = raw_output.replace(original_prompt, "").strip()
    return clean_output[:100] if clean_output else "No answer generated"


def handle_gpu_memory_error(
        error: RuntimeError,
        model: AutoModelForCausalLM,
        inputs: dict,
        device: torch.device,
        tokenizer
) -> Tuple[str, str]:
    """
    Enhanced GPU memory error handling with MPS support.
    """
    logger.warning(f"GPU memory issue: {error}")

    # Clear GPU cache based on device type
    if device.type == "mps":
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

    # CPU fallback
    logger.info("Falling back to CPU generation")
    model_cpu = model.cpu()
    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_cpu.generate(
            **inputs_cpu,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Move model back to original device
    try:
        model.to(device)
        logger.info(f"Model moved back to {device}")
    except Exception as e:
        logger.warning(f"Failed to move model back to {device}: {e}")

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
        vector_db: VectorStoreIndex,
        embedding_model: HuggingFaceEmbedding,
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
                        "similarity_method": str(similarity_method),
                        "error": None,
                        "device_used": str(device),
                        "question": prompt,
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


@track_performance("single_answer_generating")
def _generate_single_answer(prompt, model, tokenizer, device, vector_db, embedding_model, similarity_method):
    """
    Updated single answer generation with MPS safety and improved answer extraction.
    """
    try:
        # Prepare prompt with context if available
        augmented_prompt = _prepare_prompt_with_context(
            prompt, vector_db, embedding_model, similarity_method
        )

        # Use MPS-safe input preparation
        inputs = prepare_generation_inputs(augmented_prompt, tokenizer, device)

        # Use MPS-safe generation with reduced tokens to prevent repetition
        outputs = generate_text_by_device(model, inputs, device, tokenizer, max_tokens=MAX_NEW_TOKENS,
                                          temperature=TEMPERATURE)

        with monitor_performance("answer_processing"):
            outputs = outputs.cpu()
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Use improved answer extraction instead of simple split
            answer = extract_answer_from_output(raw_output, prompt)

            score = evaluate_answer_quality(answer, prompt, model, tokenizer, device)

        return {
            "answer": answer,
            "score": score,
            "error": None,
            "device_used": str(device),
            "question": prompt,
            "similarity_method": str(similarity_method),
            "raw_output": raw_output  # For debugging
        }

    except RuntimeError as e:
        # Handle GPU memory errors specifically
        if _is_gpu_memory_error(e):
            logger.warning("GPU memory error detected, attempting fallback")
            answer, device_used = handle_gpu_memory_error(e, model, inputs, device, tokenizer)
            return {
                "answer": answer,
                "score": 1.0,  # Assume fallback worked
                "error": f"GPU fallback: {str(e)}",
                "device_used": device_used,
                "question": prompt,
                "similarity_method": str(similarity_method)
            }
        else:
            raise e  # Re-raise non-memory errors
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "answer": "",
            "score": 0.0,
            "error": str(e),
            "device_used": str(device),
            "question": prompt,
            "similarity_method": str(similarity_method)
        }


@track_performance("preparing_query_context")
def _prepare_prompt_with_context(prompt, vector_db, embedding_model, similarity_method):
    """Prepare prompt with retrieved context if available."""
    original_question = prompt

    if vector_db is not None:
        with monitor_performance("context_retrieval_and_prompt_construction"):
            context = retrieve_context_with_similarity(
                original_question, vector_db, embedding_model, similarity_method=similarity_method
            )
            if context and context.strip():
                return f"Context:\n{context}\n\nQuestion: {original_question}\nAnswer:"

    return f"Question: {original_question}\nAnswer:"


def _improve_prompt(original_prompt, previous_answer, model, tokenizer, device):
    """Improve prompt based on previous answer."""
    return rephrase_query(original_prompt, previous_answer, model, tokenizer, device)


def _is_gpu_memory_error(error):
    """Enhanced GPU memory error detection for both MPS and CUDA."""
    error_str = str(error).lower()
    gpu_memory_indicators = [
        "mps",
        "out of memory",
        "bfloat16",
        "memory",
        "allocation failed",
        "cuda out of memory"
    ]
    return any(indicator in error_str for indicator in gpu_memory_indicators)


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
        tokenizer,
        device: torch.device
) -> str:
    """
    MPS-safe query rephrasing.
    """
    logger.info("Rephrasing query for improvement")

    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"Previous answer: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )

    try:
        # Use MPS-safe input preparation
        inputs = prepare_generation_inputs(rephrase_prompt, tokenizer, device, max_length=800)

        with monitor_performance("rephrase_generation"):
            with torch.no_grad():
                # Use MPS-safe generation
                outputs = generate_text_by_device(model, inputs, device, tokenizer, max_tokens=MAX_NEW_TOKENS,
                                                  temperature=TEMPERATURE)

        outputs = outputs.cpu()
        rephrased_query = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info("Query rephrased successfully")
        return rephrased_query

    except Exception as e:
        logger.error(f"Query rephrasing failed: {e}")
        # Return original prompt if rephrasing fails
        return original_prompt


# Example usage and testing
def test_mps_query_processing():
    """Test the MPS-safe query processing pipeline."""
    print.info("🧪 Testing MPS-safe query processing...")

    try:

        from utility.logger import logger
        # Load model
        tokenizer, model = load_model()
        device = next(model.parameters()).device
        logger.info(f"✅ Model loaded on {device}")

        # Load vector DB (optional)
        try:
            vector_db, embedding_model = load_vector_db(logger=logger, source="url", source_path=INDEX_SOURCE_URL)
            logger.info("✅ Vector DB loaded")
        except Exception as e:
            logger.error(f"⚠️  Vector DB failed to load: {e}")
            vector_db, embedding_model = None, None

        # Test simple query
        test_query = "What is artificial intelligence?"
        logger.info(f"\n🔍 Testing query: '{test_query}'")

        result = process_query_with_context(
            test_query, model, tokenizer, device,
            vector_db, embedding_model, max_retries=1
        )

        if result["error"]:
            logger.error(f"❌ Query failed: {result['error']}")
            return False
        else:
            logger.info(f"✅ Query successful!")
            logger.info(f"   Answer: {result['answer'][:100]}...")
            logger.info(f"   Score: {result['score']}")
            logger.info(f"   Device: {result['device_used']}")
            return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = test_mps_query_processing()
    logger.info(f"\n🎯 MPS Query Processing Test: {'PASSED' if success else 'FAILED'}")
