
# Send a query to the model.
#  Use RAG for mor accurate answer.
def query_model(prompt, model, tokenizer, device, index=None):
    # If an index is provided, retrieve context to augment the prompt.
    if index is not None:
        # Retrieve relevant documents (this returns a string with combined context)
        retrieved_context = index.query(prompt)
        # Combine retrieved context with the original prompt
        augmented_prompt = f"Context: {retrieved_context}\n\nQuestion: {prompt}\nAnswer:"
    else:
        augmented_prompt = prompt

    inputs = tokenizer(augmented_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
