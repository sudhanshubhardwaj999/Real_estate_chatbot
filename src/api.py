from fastapi import FastAPI
import requests
from .preprocess import load_and_preprocess_data, embed_data
from .retrieve import InformationRetriever

app = FastAPI()

# Load and preprocess data
data_dir = "data/"
preprocessed_text = load_and_preprocess_data(data_dir)
embeddings = embed_data(preprocessed_text)

# Initialize information retriever
retriever = InformationRetriever()
retriever.build_index(embeddings, preprocessed_text)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_API_KEY = "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"  # Replace with your actual Hugging Face API key
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def truncate_prompt(prompt, max_tokens=1024, reserved_tokens=100):
    """Truncate the prompt to ensure total tokens stay within limits."""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(prompt)

    # Ensure that total tokens including generated tokens are within the limit
    if len(tokens) > (max_tokens - reserved_tokens):
        tokens = tokens[: max_tokens - reserved_tokens]

    return tokenizer.decode(tokens, clean_up_tokenization_spaces=True)


@app.get("/query")
def query_bot(query: str):
    """API endpoint to handle user queries."""
    query_embedding = embed_data([query])
    relevant_info = retriever.query(query_embedding)
    prompt = " ".join(relevant_info)

    # Truncate the prompt based on token count
    truncated_prompt = truncate_prompt(prompt)

    response = requests.post(
        HF_API_URL, headers=headers, json={"inputs": truncated_prompt}
    )

    # Log the response for debugging
    print("Hugging Face API response:", response.json())

    # Attempt to retrieve the generated text
    try:
        generated_text = response.json()[0]["generated_text"]
    except (IndexError, KeyError) as e:
        return {"error": f"Unexpected API response format: {response.json()}"}

    return {"response": generated_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
