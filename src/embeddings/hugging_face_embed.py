
from langchain.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> list:
        return self.model.encode(query).tolist()