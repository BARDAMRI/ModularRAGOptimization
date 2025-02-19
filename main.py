from model_loader import load_model
from query import query_index


def main():
    tokenizer, model = load_model()

    user_query = input("Enter your query: ")
    response = query_index(user_query)

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()