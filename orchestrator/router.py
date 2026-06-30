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
    "code_generation": [
        # Strong authoring verbs — these unambiguously indicate code-gen intent
        # ("render a triangle", "draw a quad", "create a swapchain", ...).
        "generate", "create", "build", "render", "draw", "write", "implement",
        "scaffold", "snippet", "boilerplate",
        # Tool / language markers — code-related by default.
        "cmake", "code", "vulkan-hpp", "vulkan hpp", "vhpp", "vma",
    ],
    "debug": [
        "debug", "bug", "error", "validation", "black screen",
        "gpu hang", "hang", "crash", "regression", "broken",
        "validation layer", "spirv-val",
    ],
    "knowledge_query": [
        # Question phrases drive the textbook-question route. Vulkan nouns are
        # intentionally NOT listed here — they'd over-match "render a triangle"
        # and "create a swapchain" cases. Verbs in code_generation dominate
        # authorial intents; question phrases dominate retrospective intents.
        "retrieve", "knowledge", "docs", "doc", "spec", "specification",
        "reference", "citation", "manual", "explain",
        "what is", "what are", "how does", "why does",
    ],
    "self_update": [
        "self update", "update monitor", "spec update", "changelog",
        "khronos update",
    ],
    "platform_detect": [
        "platform", "detect", "target", "device", "adb",
        "what device", "which device", "what platform",
    ],
}


def classify_task_type(user_request: str) -> TaskType:
    """Cheap keyword-based classifier used as fallback when LLM routing fails.

    Iteration order matters: knowledge_query is checked before code_generation
    so that textbook questions about Vulkan objects ("what is a swapchain?")
    route to retrieval rather than to code generation. Strong authoring verbs
    in code_generation still win for authorial prompts ("create a swapchain").
    """
    lowered = user_request.lower()
    for task_type in ("knowledge_query", "code_generation", "platform_detect", "debug", "self_update"):
        tokens = _KEYWORD_DEFAULTS[task_type]
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


def error_node(state: Any) -> dict:
    """Terminal node — surfaces an error message back through the state channel.

    langgraph 1.x passes a typed ``VulkanMindState`` Pydantic instance to nodes
    (re-validated between edges), so we coerce and read fields as attributes
    rather than calling ``.get`` on a dict.
    """
    from orchestrator.state import coerce_state, normalize_node_return

    state_model = coerce_state(state)
    message = getattr(state_model, "error", None) or "Unknown task type; please clarify the request."
    return normalize_node_return({
        "error": message,
        "agent_trace": (state_model.agent_trace or []) + ["error_node recorded clarification request"],
    })


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
