import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model


def judge_query(query, response1, response2):
    tokenizer, model = load_model()

    prompt = f"""You are an AI judge. Compare the two responses below and choose the most accurate one.

    Example:
    Query: What is 2 + 2?
    Response 1: 5
    Response 2: 4
    Correct Answer: "Response 2"

    Now judge the following:

    Query: {query}

    Response 1: {response1}

    Response 2: {response2}

    Answer only with: "Response 1", "Response 2", or "Tie".
    """

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)  # Limit response length

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    query = "Who discovered gravity?"
    response1 = "Albert Einstein discovered gravity."
    response2 = "Isaac Newton discovered gravity."

    result = judge_query(query, response1, response2)
    print("\n🏆 Best Response:", result)
