from __future__ import annotations

import os
from typing import Any

import structlog

from agents.self_improvement.memory_injector import MemoryInjector
from llm.embedding_client import EmbeddingClient
from orchestrator.state import (
    KnowledgeChunk,
    PlatformContext,
    VulkanMindState,
    coerce_state,
    normalize_node_return,
)

from .retrieval.query_builder import build_query


class KnowledgeRetrievalResult:
    """Result container for the parallel retrieval result.

    We don't use a Pydantic BaseModel here because the dataclass fits more
    naturally with the dict-based LangGraph state object.
    """

    def __init__(
        self,
        retrieved_knowledge: list[KnowledgeChunk],
        graphify_excerpt: str | None = None,
    ) -> None:
        self.retrieved_knowledge = retrieved_knowledge
        self.graphify_excerpt = graphify_excerpt


def knowledge_retrieval_node(state: dict) -> dict:
    """LangGraph node — run Qdrant retrieval + Graphify read in parallel.

    The retrieval contract is unchanged: ``KnowledgeChunk`` list is sorted,
    graph-filtered, and emitted into ``state["retrieved_knowledge"]``. The
    Graphify excerpt (when available) lands in ``state["graphify_snapshot"]``
    which the downstream agent can pull alongside the platform context.
    """
    state_model = coerce_state(state)
    context = state_model.platform_context
    if context is None:
        return {
            "error": "platform_context is required before knowledge retrieval",
            "agent_trace": state_model.agent_trace,
        }
    runtime = _runtime()
    if runtime["memory_injector"] is not None:
        injected = inject_session_memory(state_model, runtime["memory_injector"])
        state_model = injected
    query = build_query(state_model.user_request, context, topic_hint=state_model.topic_hint)
    qdrant_chunks = _qdrant_retrieve(query)
    graphify_excerpt = _graphify_retrieve(state_model.user_request, runtime["graphify_reader"])
    return normalize_node_return({
        "session_memory": state_model.session_memory,
        "improvement_context": state_model.improvement_context,
        "retrieved_knowledge": qdrant_chunks,
        "graphify_snapshot": graphify_excerpt,
        "agent_trace": state_model.agent_trace
        + [
            f"knowledge_retrieval_node retrieved {len(qdrant_chunks)} platform-filtered chunks"
            + (" + graphify snippet" if graphify_excerpt else "")
        ],
    })


def qdrant_retrieve(
    query_text: str,
    context: PlatformContext,
    *,
    embedding_client: EmbeddingClient | None,
    query_filter,
    limit: int = 8,
    collection: str | None = None,
) -> list[KnowledgeChunk]:
    """Pure function used by `knowledge_retrieval_node` and the unit tests."""
    client = _configured_qdrant_client()
    if client is None or embedding_client is None:
        return []
    try:
        vector = embedding_client.embed(query_text)
    except Exception:
        return []
    response = client.query_points(
        collection_name=collection or os.environ.get("QDRANT_COLLECTION", "vulkanmind_chunks"),
        query=vector,
        query_filter=query_filter,
        limit=limit,
    )
    return [KnowledgeChunk.model_validate(point.payload) for point in response.points]


def _qdrant_retrieve(query) -> list[KnowledgeChunk]:
    client = _configured_qdrant_client()
    if client is None:
        return []
    embedding_client = _configured_embedding_client()
    if embedding_client is None:
        # graceful degradation: Qdrant is up but the embedder isn't configured.
        return []
    try:
        vector = embedding_client.embed(query.text)
    except Exception:
        return []
    try:
        response = client.query_points(
            collection_name=os.environ.get("QDRANT_COLLECTION", "vulkanmind_chunks"),
            query=vector,
            query_filter=query.query_filter,
            limit=query.limit,
        )
    except Exception:
        return []
    return [KnowledgeChunk.model_validate(point.payload) for point in response.points]


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


def _runtime() -> dict[str, Any]:
    """Read the runtime collaborators from the graph module.

    These collaborators (memory_injector, llm_client, graphify_reader) are
    unserialisable, so they cannot live in langgraph state — main.py wires
    them once via ``configure_self_improvement`` and nodes access them here.
    """
    from orchestrator import graph as _graph

    return _graph.get_runtime()


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


def _configured_embedding_client() -> EmbeddingClient | None:
    """Lazily resolve an EmbeddingClient from env / config.

    Kept as a module-level helper so tests can monkeypatch the loader.
    """
    from llm.embedding_client import create_embedding_client

    return create_embedding_client({})


def _graphify_retrieve(question: str, reader: Any | None) -> str | None:
    if reader is None or not question:
        return None
    try:
        return reader.format_for_prompt(question)
    except Exception:
        return None
