from llama_cpp import Llama

llm = Llama(
    model_path="models/tinyllama.gguf",
    n_ctx=2048
)

response = llm(
    "Explain machine overheating causes:",
    max_tokens=200
)

print(response["choices"][0]["text"])