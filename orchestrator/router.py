from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal[
    "code_generation",
    "debug",
    "knowledge_query",
    "platform_detect",
    "self_update",
    "unknown",
]


class _RouterDecision(BaseModel):
    task_type: TaskType
    rationale: str = Field(default="")


_KEYWORD_DEFAULTS: dict[TaskType, list[str]] = {
    "code_generation": ["generate", "create", "code", "cmake", "vulkan-hpp", "vulkan hpp"],
    "debug": ["debug", "bug", "error", "validation", "black screen", "gpu hang", "hang"],
    "knowledge_query": ["retrieve", "knowledge", "docs", "spec", "reference", "citation"],
    "self_update": ["self update", "update monitor", "spec update", "changelog"],
    "platform_detect": ["platform", "detect", "target"],
}


def classify_task_type(user_request: str) -> TaskType:
    """Cheap keyword-based classifier used as fallback when LLM routing fails."""
    lowered = user_request.lower()
    for task_type, tokens in _KEYWORD_DEFAULTS.items():
        if any(token in lowered for token in tokens):
            return task_type
    return "unknown"


def _runtime_llm_client():
    """Read the LLMClient from the graph's runtime slot.

    Tests monkeypatch ``orchestrator.graph._LLM_CLIENT`` directly; production
    paths populate it via ``configure_self_improvement``.
    """
    from orchestrator import graph as _graph

    return getattr(_graph, "_LLM_CLIENT", None)


def classify_task(
    user_request: str,
    platform_summary: str | None,
    llm_client: Any | None,
) -> _RouterDecision:
    """LLM-first task classifier.

    Behaviour:
      * If ``llm_client`` is None or completion fails, fall back to the keyword
        classifier.
      * The validator constrains task_type to the VulkanMind TaskType literal,
        so the LLM never gets to invent a new bucket.
    """
    if not user_request:
        return _RouterDecision(task_type="unknown", rationale="empty user_request")
    if llm_client is None:
        return _RouterDecision(task_type=classify_task_type(user_request), rationale="keyword fallback")
    try:
        from llm.client import LLMMessage

        system = (
            "You are the VulkanMind router. Classify the user request into exactly one task_type.\n"
            "Allowed values: code_generation, debug, knowledge_query, platform_detect, self_update, unknown.\n"
            "Return JSON matching the schema with a short rationale."
        )
        platform_block = f"\nActive PlatformContext summary:\n{platform_summary}\n" if platform_summary else ""
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(
                role="user",
                content=f"User request: {user_request}\n{platform_block}",
            ),
        ]
        decision = llm_client.complete_structured(messages, _RouterDecision, max_tokens=256)
        if decision.task_type not in {"code_generation", "debug", "knowledge_query", "platform_detect", "self_update", "unknown"}:
            return _RouterDecision(task_type=classify_task_type(user_request), rationale="LLM returned invalid enum")
        return decision
    except Exception:
        return _RouterDecision(task_type=classify_task_type(user_request), rationale="LLM failure; keyword fallback")


def error_node(state: dict) -> dict:
    message = state.get("error") or "Unknown task type; please clarify the request."
    return {
        "error": message,
        "agent_trace": (state.get("agent_trace") or []) + ["error_node recorded clarification request"],
    }


def router_node(state: dict) -> dict:
    """LangGraph router — LLM first, keyword fallback."""
    from orchestrator.state import coerce_state

    state_model = coerce_state(state)
    user_request = state_model.user_request
    platform_summary = None
    if state_model.platform_context is not None:
        try:
            from llm.prompt_builder import platform_context_summary

            platform_summary = platform_context_summary(state_model.platform_context)
        except Exception:
            platform_summary = None
    decision = classify_task(user_request, platform_summary, _runtime_llm_client())
    return {
        "task_type": decision.task_type,
        "agent_trace": state_model.agent_trace
        + [f"router_node classified task_type={decision.task_type} ({decision.rationale or 'keyword'})"],
    }
