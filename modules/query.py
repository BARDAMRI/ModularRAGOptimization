import random


def evaluate_answer_quality(answer, question, model, tokenizer, device):
    eval_prompt = (
        f"Evaluate the quality of the following answer to the question on a scale of 0 to 1.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Score:"
    )
    inputs = tokenizer(eval_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )
    score_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        score = float(score_text.split()[-1])
        return min(max(score, 0.0), 1.0)
    except ValueError:
        return 0.0


def rephrase_query(original_prompt, previous_answer, model, tokenizer, device):
    rephrase_prompt = (
        f"Original question: {original_prompt}\n"
        f"The previous answer was: {previous_answer}\n"
        f"Improve the original question to get a better answer:"
    )
    inputs = tokenizer(rephrase_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Send a query to the model with a retrieval-feedback loop.
def query_model(prompt, model, tokenizer, device, index=None, max_retries=3, quality_threshold=0.7):
    answer = ""
    score = 0.0
    attempt = 0
    current_prompt = prompt

    while attempt <= max_retries:
        if index is not None:
            retrieved_context = index.query(current_prompt)
            augmented_prompt = f"Context: {retrieved_context}\n\nQuestion: {current_prompt}\nAnswer:"
        else:
            augmented_prompt = current_prompt

        inputs = tokenizer(augmented_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = evaluate_answer_quality(answer, current_prompt, model, tokenizer, device)
        if score >= quality_threshold or attempt == max_retries:
            return answer

        current_prompt = rephrase_query(current_prompt, answer, model, tokenizer, device)
        attempt += 1

    if not answer.strip():
        return {
            "final_answer": "Model did not generate a response.",
            "final_score": 0.0,
            "attempts": attempt + 1,
            "final_prompt": current_prompt,
            "error": "Empty response from model"
        }

    return {
        "final_answer": answer,
        "final_score": score,
        "attempts": attempt + 1,
        "final_prompt": current_prompt
    }
