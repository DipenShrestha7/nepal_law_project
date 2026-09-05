from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Initialize embedding model once
model = SentenceTransformer("BAAI/bge-m3")


def build_point(point_id: int, text: str, payload_data: Dict[str, Any]) -> PointStruct:
    """Cleans null values and generates vector embeddings for Qdrant."""
    # Strip null values automatically
    clean_payload = {k: v for k, v in payload_data.items() if v is not None}
    clean_payload["content_text"] = text

    vector = model.encode(text, show_progress_bar=False).tolist()

    return PointStruct(id=point_id, vector=vector, payload=clean_payload)


def push_to_qdrant(client: QdrantClient, collection_name: str, points: list):
    """Uploads batch points to Qdrant."""
    if points:
        client.upsert(collection_name=collection_name, points=points)
