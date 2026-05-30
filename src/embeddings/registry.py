from langchain.embeddings import Embeddings
from src.config import settings

def get_embeddings() -> Embeddings:
    if settings.embedding == "ollama":
        from src.embeddings.ollama_embed import OllamaEmbeddings
        return OllamaEmbeddings(model=settings.embedding_model)
    elif settings.embedding == "huggingface":
        from src.embeddings.hugging_face_embed import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)