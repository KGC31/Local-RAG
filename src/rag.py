from functools import lru_cache
import re

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from langchain_protocol import Citation

from src.filters import filters_to_qdrant
from src.llm import invoke_llm
from src.schema import ChunkMetadata, RagAnswer, RetrievedChunk
from src.config import settings
from src.store import get_client, get_vector_store
from src.prompts.prompts import PROMPTS_DIR, ANSWER_TEMPLATE


def retrieve(query, k=None, filters=None, collection_name=None):
    hits = get_vector_store().similarity_search_with_score(
        query=query,
        k=k or settings.top_k,
        filter=filters_to_qdrant(filters),
    )

    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata)
        )
        for doc, score in hits
    ]
    
def scroll_all(collection_name, *, scroll_filter=None, batch_size=256,):
    client = get_client()

    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        yield points

        if next_offset is None:
            break

        offset = next_offset
        
    
def fetch_all_chunks(filters=None, collection_name=None):
    name = collection_name or settings.qdrant_collection

    results = []

    for page in scroll_all(
        name,
        scroll_filter=filters_to_qdrant(filters)
    ):
        for point in page:
            payload = point.payload or {}

            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""

            if meta and text:
                results.append(
                    RetrievedChunk(
                        text=text,
                        score=0.0,
                        metadata=ChunkMetadata(**meta)
                    )
                )

    return sorted(
        results,
        key=lambda r: (
            r.metadata.filename,
            r.metadata.page,
            int(r.metadata.chunk_id.rsplit(":", 1)[-1]),
        ),
    )
    
@lru_cache(maxsize=1)
def _jinja_env():
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
def render_prompt(template_name, **context):
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks):
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]
    
def build_context_text(chunks):
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata

        parts.append(
            f"[Chunk {i}] "
            f"(source={meta.filename}, "
            f"page={meta.page})\n"
            f"{chunk.text}"
        )

    return "\n\n".join(parts)

def filter_used_citations(answer_text, citations, chunks):
    used_markers = set(
        re.findall(r"\[(S\d+)\]", answer_text)
    )

    filtered_citations = []
    filtered_chunks = []

    for citation, chunk in zip(
        citations,
        chunks
    ):
        if (
            citation["source_marker"]
            in used_markers
        ):
            filtered_citations.append(
                citation
            )

            filtered_chunks.append(
                chunk
            )

    return (
        filtered_citations,
        filtered_chunks,
    )
    
def answer(question, k=None, filters=None, collection_name=None):
    try:
        chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)

        if not chunks:
            return RagAnswer(
                question=question,
                answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
            )

        prompt = render_prompt(
            ANSWER_TEMPLATE,
            question=question,
            context_text=build_context_text(chunks)
        )

        text = invoke_llm(prompt).strip()

        citations = format_citations(chunks)

        filtered_citations, filtered_chunks = (
            filter_used_citations(
                text,
                citations,
                chunks
            )
        )

        return RagAnswer(
            question=question,
            answer=text,
            citations=filtered_citations,
            chunks=filtered_chunks,
        )
        
    except Exception as e:
        return RagAnswer(
            question=question,
            answer=f"Error during processing: {str(e)}"
        )