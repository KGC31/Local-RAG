from collections import defaultdict
import hashlib
from pathlib import Path
import uuid

from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.models import PointStruct

from src.chunking.registry import get_chunker
from src.embeddings.registry import get_embeddings, get_sparse_embeddings
from src.schema import ChunkMetadata
from src.store import ensure_collection, get_client
from src.config import settings

def _document_id(path):
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def _chunk_id(doc_id, page, index):
    return f"{doc_id}:{page}:{index}"

def _load_pdf(path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)

    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1

        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }

    return pages

def index_chunks(chunks, collection_name=None):
    client = get_client()
    dense_model = get_embeddings() 
    sparse_model = get_sparse_embeddings()
    
    texts = [ chunk["text"] for chunk in chunks ] 
    dense_vectors = dense_model.embed_documents(texts) 

    sparse_vectors = list(sparse_model.embed_documents(texts) ) 

    points = [] 
    for i, chunk in enumerate(chunks): 
        sparse = sparse_vectors[i] 
        point = PointStruct( 
            id=str(uuid.uuid4()), 
            vector={ 
                "dense": dense_vectors[i], 
                "sparse": { 
                    "indices": sparse.indices, 
                    "values": sparse.values, 
                }, 
            }, 
            payload={ 
                "page_content": chunk["text"], 
                "metadata": chunk["metadata"], 
            }, 
        ) 
        
        points.append(point) 

        client.upsert( collection_name=collection_name, points=points ) 
    return len(points)

def build_chunks(pdf_paths, chunk_size=None, chunk_overlap=None, chunker=None):
    page_docs = []

    for path in pdf_paths: 
        page_docs.extend(_load_pdf(path))

    splitter = get_chunker(chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = [] 
    per_doc_counter = defaultdict(int)

    for page_doc in page_docs:
        text = page_doc.page_content 
        
        if not text or not text.strip(): 
            continue
        
        chunks = splitter.chunk(page_doc.page_content)
        
        for chunk in chunks: 
            doc_id = page_doc.metadata["document_id"] 
            idx = per_doc_counter[doc_id] 
            per_doc_counter[doc_id] += 1 
            meta = ChunkMetadata( 
                document_id=doc_id, 
                filename=page_doc.metadata["filename"], 
                source=page_doc.metadata["source"], 
                page=page_doc.metadata["page"], 
                chunk_id=_chunk_id( doc_id, page_doc.metadata["page"], idx, ),
                section=page_doc.metadata.get("section"), 
            ) 
            chunk["metadata"] = meta.model_dump() 
            
            all_chunks.append(chunk) 
            
    return all_chunks

def ingest(recreate=False, collection_name=None, chunker=None, chunk_size=None, chunk_overlap=None, pdf_paths=None):
    try:
        if collection_name is None:
            collection_name = settings.qdrant_collection
            
        pdfs = [Path(pdf_path) for pdf_path in pdf_paths]

        ensure_collection(recreate=recreate, collection_name=collection_name)

        chunks = build_chunks(
            pdfs,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        index_chunks(chunks, collection_name=collection_name)
        
        return len(chunks)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return 0