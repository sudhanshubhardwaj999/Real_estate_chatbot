import sys
import os

# Add the root directory of the project to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
import uvicorn
from src.preprocess import load_and_preprocess_data, embed_data
from src.retrieve import InformationRetriever
from src.api import query_bot

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Initialize and prepare the application on startup:
    - Load and preprocess data.
    - Build FAISS index.
    """
    global retriever
    # Load and preprocess data
    data_dir = "data/"
    preprocessed_text = load_and_preprocess_data(data_dir)

    # Embed the preprocessed data using a Sentence Transformer model
    embeddings = embed_data(preprocessed_text)

    # Initialize and build the information retriever
    retriever = InformationRetriever()
    retriever.build_index(embeddings, preprocessed_text)

    # Log that the startup is complete
    print("Application startup complete. FAISS index is built and ready for queries.")


@app.get("/query")
def query(query: str):
    """
    Handle incoming queries via the FastAPI endpoint.
    """
    return query_bot(query)


if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
