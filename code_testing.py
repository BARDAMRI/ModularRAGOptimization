# ============== Getting the type of the vector_db. ============

from llama_index.core.indices.base import BaseIndex
from configurations.config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from llama_index.core import VectorStoreIndex

# ============ Getting the type of the vector_db. ============
vector_db, embedding_space = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
print(type(vector_db))  # Prints the type of vector_db

# Example: Using isinstance() to check if it's a specific type
if isinstance(vector_db, VectorStoreIndex):
    print("vector_db is a VectorStoreIndex")
    print(type(vector_db.vector_store))  # Prints the type of the vector store
elif isinstance(vector_db, BaseIndex):
    print("vector_db is a BaseIndex")
else:
    print(f"Unknown type: {type(vector_db)}")

# Example: Using dir() to inspect the object
print(dir(vector_db))  # Lists all attributes and methods of vector_db

# ============ Getting the type of the tokenizer.============


# from transformers import AutoTokenizer, GPT2TokenizerFast
#
# # Example: Initialize tokenizer
# MODEL_PATH = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#
# # Check the type of the tokenizer
# print(type(tokenizer))  # Prints the type of tokenizer
#
# # Verify if the tokenizer is an instance of GPT2TokenizerFast
# if isinstance(tokenizer, GPT2TokenizerFast):
#     print("tokenizer is a GPT2TokenizerFast")
# elif isinstance(tokenizer, AutoTokenizer):
#     print("tokenizer is an AutoTokenizer")
# else:
#     print("Unknown tokenizer type")


# ============ Getting the type of the vector_db.============
# vector_db = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
# print(type(vector_db))  # Prints the type of vector_db
#
# # Example: Using isinstance() to check if it's a specific type
#
#
# if isinstance(vector_db, VectorStoreIndex):
#     print("vector_db is a VectorStoreIndex")
# elif isinstance(vector_db, BaseIndex):
#     print("vector_db is a BaseIndex")
# else:
#     print("Unknown type")
#
# # Example: Using dir() to inspect the object
# print(dir(vector_db))  # Lists all attributes and methods of vector_db


# ============ Generating a response with refined parameters.============
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# # Initialization
# MODEL_PATH = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prompt = "Hello, how are you?"
#
# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
# # Generate output with refined parameters
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=30,  # Limit response length
#     do_sample=True,
#     temperature=0.6,    # Reduce randomness
#     top_k=20,           # Stricter vocabulary sampling
#     top_p=0.8,          # Nucleus sampling
#     repetition_penalty=2.0,  # Penalize repetition more aggressively
#     pad_token_id=tokenizer.eos_token_id
# )
#
# # Decode and print response
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(response[0])
