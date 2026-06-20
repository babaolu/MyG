from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.code_generation import code_generation_node
from agents.debugger import debugger_node
from agents.knowledge_retrieval import knowledge_retrieval_node
from agents.platform_intelligence import platform_intelligence_node
from agents.self_update import self_update_node
from orchestrator.router import error_node, router_node
from orchestrator.state import VulkanMindState


def _route_after_platform(state: VulkanMindState | dict) -> str:
    platform_context = state.get("platform_context") if isinstance(state, dict) else state.platform_context
    return "router_node" if platform_context is not None else "error_node"


def _route_after_router(state: VulkanMindState | dict) -> str:
    task_type = state.get("task_type") if isinstance(state, dict) else state.task_type
    if task_type in {"code_generation", "debug", "knowledge_query"}:
        return "knowledge_retrieval_node"
    if task_type == "self_update":
        return "self_update_node"
    return "error_node"


def _route_after_knowledge(state: VulkanMindState | dict) -> str:
    task_type = state.get("task_type") if isinstance(state, dict) else state.task_type
    if task_type == "code_generation":
        return "code_generation_node"
    if task_type == "debug":
        return "debugger_node"
    return END


def _route_after_code_generation(state: VulkanMindState | dict) -> str:
    validation_passed = state.get("validation_passed") if isinstance(state, dict) else state.validation_passed
    return END if validation_passed else "debugger_node"


def build_graph(database_path: str | Path = "data/vulkanmind.sqlite3") -> StateGraph:
    builder = StateGraph(VulkanMindState)
    builder.add_node("platform_intelligence_node", platform_intelligence_node)
    builder.add_node("router_node", router_node)
    builder.add_node("knowledge_retrieval_node", knowledge_retrieval_node)
    builder.add_node("code_generation_node", code_generation_node)
    builder.add_node("debugger_node", debugger_node)
    builder.add_node("self_update_node", self_update_node)
    builder.add_node("error_node", error_node)
    builder.add_edge("START", "platform_intelligence_node")
    builder.add_conditional_edges("platform_intelligence_node", _route_after_platform)
    builder.add_conditional_edges("router_node", _route_after_router)
    builder.add_conditional_edges("knowledge_retrieval_node", _route_after_knowledge)
    builder.add_conditional_edges("code_generation_node", _route_after_code_generation)
    builder.add_edge("debugger_node", END)
    builder.add_edge("self_update_node", END)
    builder.add_edge("error_node", END)
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{Path(database_path).resolve()}")
    return builder.compile(checkpointer=checkpointer)


def create_app_graph(database_path: str | Path = "data/vulkanmind.sqlite3") -> StateGraph:
    return build_graph(database_path)
