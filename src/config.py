from functools import lru_cache
from pathlib import Path
from typing_extensions import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / "backend/.env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    # --- Chunking & retrieval ---
    chunking_strategy: Literal["recursive", "semantic"] = "semantic"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    # --- Embeddings & reranker ---
    embedding: Literal["ollama", "huggingface"] = "ollama"
    embedding_model: str = "qwen3-embedding:8b"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        
        if self.embedding not in ["ollama", "huggingface"]:
            raise ValueError("embedding must be either 'ollama' or 'huggingface'.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()