from src.chunking.base import BaseChunker

def get_chunker(chunker: str = "recursive",**kwargs) -> BaseChunker:
    # Recursive chunking
    if chunker == "recursive":
        from src.chunking.chunking_strategies.recursive_chunking import RecursiveChunker

        config = {}

        if kwargs.get("chunk_size") is not None:
            config["chunk_size"] = kwargs["chunk_size"]
            
        if kwargs.get("chunk_overlap") is not None:
            config["chunk_overlap"] = kwargs["chunk_overlap"]

        return RecursiveChunker(**config)

    # Semantic chunking
    elif chunker == "semantic":
        from src.chunking.chunking_strategies.semantic_chunking import SemanticChunker

        config = {}

        if kwargs.get("embedding_model") is not None:
            config["embedding_model"] = kwargs["embedding_model"]

        if kwargs.get("similarity_threshold") is not None:
            config["similarity_threshold"] = kwargs["similarity_threshold"]

        if kwargs.get("min_sentences_per_chunk") is not None:
            config["min_sentences_per_chunk"] = kwargs["min_sentences_per_chunk"]

        if kwargs.get("max_chunk_size") is not None:
            config["max_chunk_size"] = kwargs["max_chunk_size"]

        return SemanticChunker(**config)

    raise ValueError(f"Unsupported chunker: {chunker}")