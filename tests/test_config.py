from types import NoneType
from scripts.evaluator import load_llm, load_embedding_model, run_llm_query
from config import HF_MODEL_NAME, LLM_MODEL_NAME


def test_configuration_with_sample_query() -> NoneType:
    """
    Tests the configuration with a sample query.

    Returns:
        NoneType: No return value.
    """
    print("\nTesting configuration with a sample query...")

    # Load lightweight LLM and embedding model
    print("\n> Loading LLM...")
    tokenizer, model = load_llm()
    print(f"LLM Model Loaded: {LLM_MODEL_NAME}")

    print("\n> Loading Embedding Model...")
    embedding_model = load_embedding_model()
    print(f"Embedding Model Loaded: {HF_MODEL_NAME}")

    # Run a sample query
    sample_query: str = "What is the capital of France?"
    print(f"\n> Running sample query: {sample_query}")
    result = run_llm_query(sample_query, tokenizer, model)

    print("\n> Query Result:")
    print(result)


if __name__ == "__main__":
    test_configuration_with_sample_query()
