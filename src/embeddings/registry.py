
from langchain.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse
from src.config import settings

def get_embeddings() -> Embeddings:
    if settings.embedding == "ollama":
        from src.embeddings.ollama_embed import OllamaEmbeddings
        return OllamaEmbeddings(model=settings.embedding_model)
    elif settings.embedding == "huggingface":
        from src.embeddings.hugging_face_embed import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
def get_sparse_embeddings() -> Embeddings:
    return FastEmbedSparse(model_name="Qdrant/bm25")