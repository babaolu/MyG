from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.http import models

from orchestrator.state import PlatformContext


@dataclass(frozen=True)
class QdrantQuery:
    text: str
    query_filter: models.Filter
    limit: int = 8


def build_query(user_question: str, platform_context: PlatformContext, topic_hint: str | None = None) -> QdrantQuery:
    gpu_vendor = platform_context.target.gpu_vendor
    vulkan_version = f"vulkan_{platform_context.target.vulkan_version.replace('.', '_')}"
    must = [
        models.FieldCondition(key="platform_tags", match=models.MatchValue(value=gpu_vendor.lower())),
        models.FieldCondition(key="vulkan_version", match=models.MatchValue(value=vulkan_version)),
    ]
    should = [
        models.FieldCondition(key="platform_tags", match=models.MatchValue(value=platform_context.target.os.lower())),
        models.FieldCondition(key="platform_tags", match=models.MatchValue(value=vulkan_version)),
    ]
    if topic_hint:
        should.append(models.FieldCondition(key="topic_tags", match=models.MatchValue(value=topic_hint.lower())))
    return QdrantQuery(
        text=user_question,
        query_filter=models.Filter(must=must, should=should),
        limit=8,
    )
