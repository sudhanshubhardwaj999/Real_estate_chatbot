import faiss
import numpy as np


class InformationRetriever:
    def __init__(self):
        self.index = None
        self.data = []

    def build_index(self, embeddings, data):
        """Build FAISS index from embeddings."""
        if embeddings.ndim != 2:
            raise ValueError("Embeddings should be a 2D array")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.data = data

    def query(self, query_embedding):
        """Retrieve relevant information based on query embedding."""
        if query_embedding.ndim != 2:
            raise ValueError("Query embedding should be a 2D array")

        D, I = self.index.search(query_embedding, k=5)  # Return top 5 results
        results = [self.data[i] for i in I[0]]
        return results
