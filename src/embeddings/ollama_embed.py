from typing import List

import ollama
from langchain.embeddings import Embeddings

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_query(self, query: str) -> list:
        response = ollama.embeddings(model=self.model, prompt=query)
        return response['embedding']
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = ollama.embed(
            model=self.model,
            input=texts
        )

        return response["embeddings"]
