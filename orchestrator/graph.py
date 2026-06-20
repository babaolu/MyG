from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.code_generation import code_generation_node
from agents.debugger import debugger_node, post_session_record
from agents.debugger.pattern_library import configure_pattern_store
from agents.knowledge_retrieval import knowledge_retrieval_node
from agents.platform_intelligence import platform_intelligence_node
from agents.self_improvement.memory_injector import MemoryInjector
from agents.self_improvement.skill_extractor import SkillExtractor
from agents.self_update import self_update_node
from db.execution_traces import ExecutionTraceStore
from db.skill_writebacks import SkillStore
from orchestrator.router import error_node, router_node
from orchestrator.state import VulkanMindState

_TRACE_STORE: ExecutionTraceStore | None = None
_SKILL_STORE: SkillStore | None = None
_MEMORY_INJECTOR: MemoryInjector | None = None
_SKILL_EXTRACTOR: SkillExtractor | None = None
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def configure_self_improvement(
    database_path: str | Path,
    trace_store: ExecutionTraceStore | None = None,
    skill_store: SkillStore | None = None,
    memory_injector: MemoryInjector | None = None,
    skill_extractor: SkillExtractor | None = None,
) -> None:
    global _TRACE_STORE, _SKILL_STORE, _MEMORY_INJECTOR, _SKILL_EXTRACTOR
    _TRACE_STORE = trace_store or ExecutionTraceStore(str(database_path))
    _SKILL_STORE = skill_store or SkillStore(str(database_path))
    configure_pattern_store(_SKILL_STORE)
    _MEMORY_INJECTOR = memory_injector or MemoryInjector(_SKILL_STORE, _TRACE_STORE)
    _SKILL_EXTRACTOR = skill_extractor or SkillExtractor(_anthropic_client(), _SKILL_STORE)


def get_self_improvement_components() -> tuple[ExecutionTraceStore, SkillStore, MemoryInjector, SkillExtractor]:
    if _TRACE_STORE is None or _SKILL_STORE is None or _MEMORY_INJECTOR is None or _SKILL_EXTRACTOR is None:
        raise RuntimeError("self-improvement components are not configured")
    return _TRACE_STORE, _SKILL_STORE, _MEMORY_INJECTOR, _SKILL_EXTRACTOR


def self_improvement_node(state: VulkanMindState | dict) -> dict:
    state_model = _state_from_mapping(state)
    if state_model.self_improvement_phase == "start":
        if state_model.platform_context is None:
            return {"error": "platform_context is required before memory injection"}
        memory = _MEMORY_INJECTOR.build_session_memory(state_model.platform_context, state_model.task_type)
        injected = _MEMORY_INJECTOR.inject_into_state(state_model, memory)
        return {
            "session_memory": injected.session_memory,
            "improvement_context": injected.improvement_context,
            "self_improvement_phase": "start",
            "agent_trace": state_model.agent_trace + ["self_improvement_node injected accumulated knowledge"],
        }
    if state_model.self_improvement_phase == "end":
        trace_id = state_model.trace_id or str(uuid4())
        state_model.trace_id = trace_id
        _EXECUTOR.submit(_record_trace_async, state_model)
        return {
            "trace_id": trace_id,
            "self_improvement_phase": "end",
            "agent_trace": state_model.agent_trace + ["self_improvement_node scheduled execution trace recording"],
        }
    return {"agent_trace": state_model.agent_trace}


def _record_trace_async(state: VulkanMindState) -> None:
    if _TRACE_STORE is None or _SKILL_EXTRACTOR is None:
        return
    post_session_record(state, _TRACE_STORE, _SKILL_EXTRACTOR)


def _route_after_platform(state: VulkanMindState | dict) -> str:
    platform_context = state.get("platform_context") if isinstance(state, dict) else state.platform_context
    return "self_improvement_node" if platform_context is not None else "error_node"


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
    return "self_improvement_node" if validation_passed else "debugger_node"


def _route_after_self_improvement(state: VulkanMindState | dict) -> str:
    phase = state.get("self_improvement_phase") if isinstance(state, dict) else state.self_improvement_phase
    return "router_node" if phase == "start" else END


def build_graph(database_path: str | Path = "data/vulkanmind.sqlite3") -> StateGraph:
    configure_self_improvement(database_path)
    builder = StateGraph(VulkanMindState)
    builder.add_node("platform_intelligence_node", platform_intelligence_node)
    builder.add_node("self_improvement_node", self_improvement_node)
    builder.add_node("router_node", router_node)
    builder.add_node("knowledge_retrieval_node", knowledge_retrieval_node)
    builder.add_node("code_generation_node", code_generation_node)
    builder.add_node("debugger_node", debugger_node)
    builder.add_node("self_update_node", self_update_node)
    builder.add_node("error_node", error_node)
    builder.add_edge("START", "platform_intelligence_node")
    builder.add_conditional_edges("platform_intelligence_node", _route_after_platform)
    builder.add_conditional_edges("self_improvement_node", _route_after_self_improvement)
    builder.add_conditional_edges("router_node", _route_after_router)
    builder.add_conditional_edges("knowledge_retrieval_node", _route_after_knowledge)
    builder.add_conditional_edges("code_generation_node", _route_after_code_generation)
    builder.add_edge("debugger_node", "self_improvement_node")
    builder.add_edge("self_update_node", END)
    builder.add_edge("error_node", END)
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{Path(database_path).resolve()}")
    return builder.compile(checkpointer=checkpointer)


def create_app_graph(database_path: str | Path = "data/vulkanmind.sqlite3") -> StateGraph:
    return build_graph(database_path)


def _state_from_mapping(state: VulkanMindState | dict) -> VulkanMindState:
    if isinstance(state, VulkanMindState):
        return state
    return VulkanMindState.model_validate(state)


def _anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)
