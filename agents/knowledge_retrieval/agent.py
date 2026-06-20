from __future__ import annotations

import os
from typing import Any

import httpx
import structlog
from pydantic import BaseModel

from agents.self_improvement.memory_injector import MemoryInjector
from orchestrator.state import KnowledgeChunk, PlatformContext, VulkanMindState

from .retrieval.query_builder import build_query


class KnowledgeRetrievalResult(BaseModel):
    retrieved_knowledge: list[KnowledgeChunk]


class EmbeddingClient:
    def embed(self, text: str) -> list[float]:
        provider = os.environ.get("EMBEDDING_PROVIDER", "local").lower()
        if provider == "openai":
            return self._openai_embed(text)
        return [float((ord(char) % 17) / 17.0) for char in text[:1536]].ljust(1536, 0.0)

    def _openai_embed(self, text: str) -> list[float]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "text-embedding-3-large", "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def knowledge_retrieval_node(state: dict) -> dict:
    context = state.get("platform_context")
    if context is None:
        return {"error": "platform_context is required before knowledge retrieval", "agent_trace": state.get("agent_trace", [])}
    if isinstance(context, dict):
        context = PlatformContext.model_validate(context)
    memory_injector = state.get("memory_injector")
    if memory_injector is not None:
        injected_state = inject_session_memory(_state_from_mapping(state), memory_injector)
        state["session_memory"] = injected_state.session_memory
        state["improvement_context"] = injected_state.improvement_context
    query = build_query(state.get("user_request", ""), context, topic_hint=state.get("topic_hint"))
    client = _configured_qdrant_client()
    if client is None:
        return {
            "retrieved_knowledge": [],
            "agent_trace": state.get("agent_trace", []) + ["knowledge_retrieval_node skipped: Qdrant unavailable"],
        }
    vector = EmbeddingClient().embed(query.text)
    response = client.query_points(
        collection_name=os.environ.get("QDRANT_COLLECTION", "vulkanmind_chunks"),
        query=vector,
        query_filter=query.query_filter,
        limit=query.limit,
    )
    chunks = [KnowledgeChunk.model_validate(point.payload) for point in response.points]
    return {
        "retrieved_knowledge": chunks,
        "agent_trace": state.get("agent_trace", []) + ["knowledge_retrieval_node retrieved platform-filtered chunks"],
    }


def inject_session_memory(
    state: VulkanMindState,
    memory_injector: MemoryInjector,
) -> VulkanMindState:
    memory = memory_injector.build_session_memory(
        platform_context=state.platform_context,
        task_type=state.task_type,
    )
    state = memory_injector.inject_into_state(state, memory)
    structlog.get_logger("vulkanmind.self_improvement.memory_injection").info(
        "session_memory_injected",
        skills=len(memory.relevant_skills),
        recent_fixes=len(memory.recent_fixes),
    )
    return state


def _state_from_mapping(state: dict) -> VulkanMindState:
    if isinstance(state, VulkanMindState):
        return state
    return VulkanMindState.model_validate(state)


def _configured_qdrant_client() -> Any | None:
    try:
        from qdrant_client import QdrantClient

        return QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            timeout=5,
        )
    except Exception:
        return None
