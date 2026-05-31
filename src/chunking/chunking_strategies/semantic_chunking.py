from typing import List, Dict
import numpy as np
import spacy

from sentence_transformers import SentenceTransformer

from src.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en",
        similarity_threshold: float = 0.75,
        min_sentences_per_chunk: int = 2,
        max_chunk_size: int = 1200,
    ):
        super().__init__()

        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_chunk_size = max_chunk_size

        self.nlp = spacy.load("en_core_web_sm")

        self.embedder = SentenceTransformer(
            embedding_model
        )

    def cosine_similarity(self, a, b) -> float:
        return np.dot(a, b) / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

    def split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        
        return [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip()
        ]

    def should_create_boundary(self, similarity: float, current_chunk: List[str], current_length: int) -> bool:
        semantic_break = similarity < self.similarity_threshold
        size_limit = current_length >= self.max_chunk_size
        enough_sentences = len(current_chunk) >= self.min_sentences_per_chunk

        return (semantic_break and enough_sentences) or size_limit

    def chunk(self, text: str) -> List[Dict]:
        if not text.strip():
            return []

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        embeddings = self.embedder.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):

            similarity = (
                self.cosine_similarity(
                    embeddings[i - 1],
                    embeddings[i],
                )
            )

            if self.should_create_boundary(
                similarity=similarity,
                current_chunk=current_chunk,
                current_length=current_length,
            ):

                chunk_text = " ".join(current_chunk)

                chunks.append(
                    self._build_chunk_metadata(
                        text=chunk_text,
                        chunk_id=f"semantic_{len(chunks)}",
                        strategy="semantic",
                    )
                )

                current_chunk = []
                current_length = 0

            current_chunk.append(sentences[i])

            current_length += len(sentences[i])

        if current_chunk:
            chunk_text = " ".join(current_chunk)

            chunks.append(
                self._build_chunk_metadata(
                    text=chunk_text,
                    chunk_id=f"semantic_{len(chunks)}",
                    strategy="semantic",
                )
            )

        return chunks