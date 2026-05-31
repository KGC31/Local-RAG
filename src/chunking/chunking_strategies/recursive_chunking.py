from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.base import BaseChunker

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, separators: List[str] | None = None,):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def chunk(self, text: str) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=False,
        )

        chunks = splitter.split_text(text)

        return [self._build_chunk_metadata(chunk, f"chunk_{i}", "recursive") for i, chunk in enumerate(chunks)]