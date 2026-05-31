from abc import ABC, abstractmethod
from typing import List, Dict

class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str) -> List[Dict]:
        pass

    def _build_chunk_metadata(self, text: str, chunk_id: str, strategy: str) -> Dict:
        return {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {
                "chunking_strategy": strategy,
                "token_count": len(text.split()),
            },
        }