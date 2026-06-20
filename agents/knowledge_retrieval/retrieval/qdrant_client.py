from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class QdrantQuery:
    query_filter: models.Filter
    limit: int = 8


def connect(host: str = "localhost", port: int = 6333) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536) -> None:
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[Any], embeddings: list[list[float]]) -> None:
    points = [
        models.PointStruct(
            id=chunk.id,
            vector=vector,
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
