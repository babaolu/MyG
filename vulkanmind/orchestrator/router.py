from __future__ import annotations

from typing import Literal


def classify_task_type(user_request: str) -> Literal["code_generation", "debug", "knowledge_query", "platform_detect", "self_update", "unknown"]:
    lowered = user_request.lower()
    if any(token in lowered for token in ["generate", "create", "code", "cmake", "vulkan-hpp", "vulkan hpp"]):
        return "code_generation"
    if any(token in lowered for token in ["debug", "bug", "error", "validation", "black screen", "gpu hang", "hang"]):
        return "debug"
    if any(token in lowered for token in ["retrieve", "knowledge", "docs", "spec", "reference", "citation"]):
        return "knowledge_query"
    if any(token in lowered for token in ["self update", "update monitor", "spec update", "changelog"]):
        return "self_update"
    if any(token in lowered for token in ["platform", "detect", "target"]):
        return "platform_detect"
    return "unknown"


def router_node(state: dict) -> dict:
    task_type = classify_task_type(state.get("user_request", ""))
    trace = state.get("agent_trace", []) + [f"router_node classified task_type={task_type}"]
    return {"task_type": task_type, "agent_trace": trace}


def error_node(state: dict) -> dict:
    message = state.get("error") or "Unknown task type; please clarify the request."
    return {"error": message, "agent_trace": state.get("agent_trace", []) + ["error_node recorded clarification request"]}
