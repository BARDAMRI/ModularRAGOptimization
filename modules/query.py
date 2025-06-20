# query.py
from functools import lru_cache
import numpy as np
from typing import Union, Optional, List, Dict
from llama_index.core import VectorStoreIndex
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import heapq
from difflib import SequenceMatcher

from transformers import AutoModelForCausalLM, GPT2TokenizerFast

from config import MAX_RETRIES, QUALITY_THRESHOLD, MAX_NEW_TOKENS
from utility.embedding_calculations import get_text_embedding
from utility.logger import logger  # Import logger


def score_similarity(text: str, query: Union[str, np.ndarray]) -> float:
    logger.info("Calculating similarity score between text and query.")
    if isinstance(query, str):
        score = SequenceMatcher(None, text.lower(), query.lower()).ratio()
        logger.info(f"Similarity score calculated: {score}")
        return score
    else:
        logger.error("score_similarity only supports string queries, not vectors.")
        raise TypeError("score_similarity only supports string queries, not vectors.")


def vector_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    logger.info("Calculating cosine similarity between two vectors.")
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    logger.info(f"Cosine similarity calculated: {similarity}")
    return similarity


@lru_cache(maxsize=1000)
def get_cached_embedding(text: str, embed_model: HuggingFaceEmbedding) -> np.ndarray:
    logger.info(f"Retrieving cached embedding for text: {text[:30]}...")
    embedding = get_text_embedding(text, embed_model)
    logger.info("Cached embedding retrieved successfully.")
    return embedding


def retrieve_context(
        query: Union[str, np.ndarray],
        vector_db: VectorStoreIndex,
        embed_model: Optional[HuggingFaceEmbedding] = None,
        top_k: int = 5,
        similarity_cutoff: float = 0.5,
        convert_to_vector: bool = False
) -> str:
    logger.info("Retrieving context for the query.")
    if not isinstance(query, (str, np.ndarray)):
        logger.error("Query must be a string or a numpy.ndarray.")
        raise TypeError("Query must be a string or a numpy.ndarray.")

    retriever = vector_db.as_retriever()
    nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(nodes)} nodes from vector database.")

    query_vector: Optional[np.ndarray] = None
    if isinstance(query, str) and convert_to_vector:
        if embed_model is None:
            logger.error("embed_model is required for converting string queries to vectors.")
            raise ValueError("embed_model is required for converting string queries to vectors.")
        query_vector = get_text_embedding(query, embed_model)
    elif isinstance(query, np.ndarray):
        query_vector = query

    contents: List[str] = [node.get_content() for node in nodes]
    document_embeddings: Optional[np.ndarray] = None
    if convert_to_vector:
        document_embeddings = np.array([get_cached_embedding(content, embed_model) for content in contents])

    similarity_scores: Union[np.ndarray, List[float]]
    if query_vector is not None and document_embeddings is not None:
        similarity_scores = np.dot(document_embeddings, query_vector) / (
                np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_vector)
        )
    else:
        similarity_scores = [score_similarity(content, query) for content in contents]

    scored_nodes = zip(similarity_scores, contents)
    top_nodes = heapq.nlargest(top_k, scored_nodes, key=lambda x: x[0])
    filtered_nodes = [content for score, content in top_nodes if score >= similarity_cutoff]

    logger.info(f"Filtered {len(filtered_nodes)} nodes based on similarity cutoff.")
    return "\n".join(filtered_nodes)


def evaluate_answer_quality(
        answer: str,
        question: str,
        model: AutoModelForCausalLM,
        tokenizer: GPT2TokenizerFast,
        device: torch.device
) -> float:
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
    logger.info("Rephrasing the query based on the previous answer.")
    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"The previous answer was: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )
    inputs = tokenizer(rephrase_prompt, return_tensors="pt").to(device)
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
        max_retries: int = MAX_RETRIES,
        quality_threshold: float = QUALITY_THRESHOLD
) -> Dict[str, Union[str, float, int, None]]:
    logger.info("Starting query process with the model.")
    try:
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model

        answer: str = ""
        score: float = 0.0
        attempt: int = 0
        current_prompt: str = prompt

        while attempt <= max_retries:
            try:
                if vector_db is not None:
                    retrieved_context = retrieve_context(current_prompt, vector_db)
                    augmented_prompt = f"Context: {retrieved_context}\n\nQuestion: {current_prompt}\nAnswer:"
                else:
                    augmented_prompt = current_prompt

                inputs = tokenizer(augmented_prompt, return_tensors="pt").to(device)
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
