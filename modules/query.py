# query.py

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from difflib import SequenceMatcher
from config import MAX_RETRIES, QUALITY_THRESHOLD, RETRIEVER_TOP_K, SIMILARITY_CUTOFF, MAX_NEW_TOKENS


def score_similarity(text, query):
    return SequenceMatcher(None, text.lower(), query.lower()).ratio()


def retrieve_context(query, index, top_k=RETRIEVER_TOP_K, similarity_cutoff=SIMILARITY_CUTOFF):
    retriever = index.as_retriever()
    retriever.postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
    nodes = retriever.retrieve(query)
    top_nodes = sorted(nodes, key=lambda n: score_similarity(n.get_content(), query), reverse=True)[:top_k]
    return "\n".join([node.get_content() for node in top_nodes])


def evaluate_answer_quality(answer, question, model, tokenizer, device):
    return 1.0
    # eval_prompt = (
    #     f"Evaluate the quality of the following answer to the question on a scale of 0 to 1.\n\n"
    #     f"Question: {question}\n"
    #     f"Answer: {answer}\n"
    #     f"Score:"
    # )
    # inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=MAX_NEW_TOKENS,
    #     do_sample=False
    # )
    # score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # try:
    #     score = float(score_text.split()[-1])
    #     return min(max(score, 0.0), 1.0)
    # except ValueError:
    #     return 0.0


def rephrase_query(original_prompt, previous_answer, model, tokenizer, device):
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Send a query to the model with a retrieval-feedback loop.
def query_model(prompt, model, tokenizer, device, index=None, max_retries=MAX_RETRIES,
                quality_threshold=QUALITY_THRESHOLD):
    try:
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model

        answer = ""
        score = 0.0
        attempt = 0
        current_prompt = prompt

        while attempt <= max_retries:
            try:
                if index is not None:
                    retrieved_context = retrieve_context(current_prompt, index)
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
                return {
                    "question": prompt,
                    "answer": answer,
                    "score": score,
                    "attempts": attempt + 1,
                    "error": str(err)
                }

        if not answer.strip():
            return {
                "question": prompt,
                "answer": "Model did not generate a response.",
                "score": 0.0,
                "attempts": attempt + 1,
                "error": "Empty response from model"
            }

        return {
            "final_answer": answer,
            "final_score": score,
            "attempts": attempt + 1,
            "final_prompt": current_prompt
        }
    except ValueError as err:
        return {
            "question": prompt,
            "answer": "Error during generation.",
            "score": 0.0,
            "attempts": 0,
            "error": str(err)
        }
