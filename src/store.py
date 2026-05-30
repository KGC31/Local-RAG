from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from src.config import settings

client = QdrantClient(path=settings.storage_dir)

def get_client() -> QdrantClient:
    return client

def create_collection():
    if settings.qdrant_collection not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )