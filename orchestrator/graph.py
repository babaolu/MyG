from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

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
from orchestrator.state import VulkanMindState, coerce_state, normalize_node_return
from utils.config import load_config

_TRACE_STORE: ExecutionTraceStore | None = None
_SKILL_STORE: SkillStore | None = None
_MEMORY_INJECTOR: MemoryInjector | None = None
_SKILL_EXTRACTOR: SkillExtractor | None = None
# Runtime collaborators pushed into nodes without round-tripping them through
# langgraph state (they are unserialisable, so they'd break the msgpack
# checkpoint path). main.py wires these in via ``configure_self_improvement``
# and nodes read them back via ``get_runtime``.
_LLM_CLIENT: Any | None = None
_GRAPHIFY_READER: Any | None = None
_GRAPHIFY_CONFIG: dict[str, Any] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def configure_self_improvement(
    database_path: str | Path,
    trace_store: ExecutionTraceStore | None = None,
    skill_store: SkillStore | None = None,
    memory_injector: MemoryInjector | None = None,
    skill_extractor: SkillExtractor | None = None,
    llm_client: Any | None = None,
    graphify_reader: Any | None = None,
) -> None:
    """Bootstrap the self-improvement subsystem + runtime collaborators.

    When ``skill_extractor`` is omitted we lazily build one from the loaded
    config — this preserves the historical "configure-then-use" pattern while
    keeping the LLMClient construction out of the call site.
    """
    global _TRACE_STORE, _SKILL_STORE, _MEMORY_INJECTOR, _SKILL_EXTRACTOR
    global _LLM_CLIENT, _GRAPHIFY_READER
    _TRACE_STORE = trace_store or ExecutionTraceStore(str(database_path))
    _SKILL_STORE = skill_store or SkillStore(str(database_path))
    configure_pattern_store(_SKILL_STORE)
    _MEMORY_INJECTOR = memory_injector or MemoryInjector(_SKILL_STORE, _TRACE_STORE)
    # ``_LLM_CLIENT`` and ``_GRAPHIFY_READER`` are runtime collaborators that
    # may already be populated by an earlier call (e.g. ``main.py`` first
    # configures them with concrete objects, then calls ``build_graph`` which
    # would otherwise re-enter here with default args and silently clobber
    # them back to ``None``). Treat each parameter as authoritative only when
    # explicitly provided, so a follow-up call with all-default args is a
    # no-op rather than an overwrite.
    if llm_client is not None:
        _LLM_CLIENT = llm_client
    if graphify_reader is not None:
        _GRAPHIFY_READER = graphify_reader
    if skill_extractor is not None:
        _SKILL_EXTRACTOR = skill_extractor
    else:
        from llm.factory import create_llm_client

        _SKILL_EXTRACTOR = SkillExtractor(create_llm_client(load_config().get("llm")), _SKILL_STORE)


def get_self_improvement_components() -> tuple[ExecutionTraceStore, SkillStore, MemoryInjector, SkillExtractor]:
    if _TRACE_STORE is None or _SKILL_STORE is None or _MEMORY_INJECTOR is None or _SKILL_EXTRACTOR is None:
        raise RuntimeError("self-improvement components are not configured")
    return _TRACE_STORE, _SKILL_STORE, _MEMORY_INJECTOR, _SKILL_EXTRACTOR


def get_runtime() -> dict[str, Any]:
    """Runtime collaborators injected via ``configure_self_improvement``.

    These live outside the langgraph state channels because they include
    unserialisable objects (MemoryInjector, LLMClient, GraphifyReader). Nodes
    read them via this accessor; tests can monkeypatch the module globals
    directly.
    """
    return {
        "memory_injector": _MEMORY_INJECTOR,
        "llm_client": _LLM_CLIENT,
        "graphify_reader": _GRAPHIFY_READER,
    }


def set_graphify_path(path: str | Path) -> bool:
    """Switch the Graphify graph to a different path at runtime.

    Returns ``True`` if the new path was loaded successfully, ``False`` if
    the file does not exist. Updates ``_GRAPHIFY_READER`` atomically.
    """
    global _GRAPHIFY_READER, _GRAPHIFY_CONFIG
    graph_path = Path(path)
    if not graph_path.exists():
        return False
    from tools.graphify_reader import GraphifyReader

    try:
        _GRAPHIFY_READER = GraphifyReader(graph_path=graph_path)
        _GRAPHIFY_CONFIG["graph_path"] = str(graph_path)
        return True
    except Exception as e:
        import sys
        print(f"[set_graphify_path] error: {e}", file=sys.stderr)
        return False


def self_improvement_node(state: VulkanMindState | dict) -> dict:
    state_model = coerce_state(state)
    if state_model.self_improvement_phase == "start":
        if state_model.platform_context is None:
            return normalize_node_return({
                "error": "platform_context is required before memory injection",
                "agent_trace": state_model.agent_trace,
            })
        # ``session_memory`` is populated by the start branch itself. If we
        # re-enter with phase=start after the user's task has already routed
        # back through us, that means we've completed one full workflow —
        # fast-forward to end so the routing decision terminates.
        if state_model.session_memory is not None:
            return normalize_node_return({
                "self_improvement_phase": "end",
                "agent_trace": state_model.agent_trace + ["self_improvement_node skipped re-injection (memory already present)"],
            })
        memory = _MEMORY_INJECTOR.build_session_memory(state_model.platform_context, state_model.task_type)
        injected = _MEMORY_INJECTOR.inject_into_state(state_model, memory)
        return normalize_node_return({
            "session_memory": injected.session_memory,
            "improvement_context": injected.improvement_context,
            # Stay in ``start`` phase on first visit so conditional edges route
            # us to ``router_node``; subsequent visits (with session_memory
            # already populated) flip to ``end`` to terminate the graph.
            "self_improvement_phase": "start",
            "agent_trace": state_model.agent_trace + ["self_improvement_node injected accumulated knowledge"],
        })
    if state_model.self_improvement_phase == "end":
        trace_id = state_model.trace_id or str(uuid4())
        state_model.trace_id = trace_id
        _EXECUTOR.submit(_record_trace_async, state_model)
        return normalize_node_return({
            "trace_id": trace_id,
            "self_improvement_phase": "end",
            "agent_trace": state_model.agent_trace + ["self_improvement_node scheduled execution trace recording"],
        })
    return normalize_node_return({"agent_trace": state_model.agent_trace})


def _record_trace_async(state: VulkanMindState) -> None:
    if _TRACE_STORE is None or _SKILL_EXTRACTOR is None:
        return
    post_session_record(state, _TRACE_STORE, _SKILL_EXTRACTOR)


def _route_after_platform(state: VulkanMindState | dict) -> str:
    state_model = coerce_state(state)
    return "self_improvement_node" if state_model.platform_context is not None else "error_node"


def _route_after_router(state: VulkanMindState | dict) -> str:
    state_model = coerce_state(state)
    task_type = state_model.task_type
    if task_type in {"code_generation", "debug", "knowledge_query"}:
        return "knowledge_retrieval_node"
    if task_type == "self_update":
        return "self_update_node"
    return "error_node"


def _route_after_knowledge(state: VulkanMindState | dict) -> str:
    state_model = coerce_state(state)
    task_type = state_model.task_type
    if task_type == "code_generation":
        return "code_generation_node"
    if task_type == "debug":
        return "debugger_node"
    return END


def _route_after_code_generation(state: VulkanMindState | dict) -> str:
    state_model = coerce_state(state)
    return "self_improvement_node" if state_model.validation_passed else "debugger_node"


def _route_after_self_improvement(state: VulkanMindState | dict) -> str:
    state_model = coerce_state(state)
    return "router_node" if state_model.self_improvement_phase == "start" else END


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
    builder.add_edge(START, "platform_intelligence_node")
    builder.add_conditional_edges("platform_intelligence_node", _route_after_platform)
    builder.add_conditional_edges("self_improvement_node", _route_after_self_improvement)
    builder.add_conditional_edges("router_node", _route_after_router)
    builder.add_conditional_edges("knowledge_retrieval_node", _route_after_knowledge)
    builder.add_conditional_edges("code_generation_node", _route_after_code_generation)
    builder.add_edge("debugger_node", "self_improvement_node")
    builder.add_edge("self_update_node", END)
    builder.add_edge("error_node", END)
    checkpointer_conn = sqlite3.connect(str(Path(database_path).resolve()), check_same_thread=False)
    checkpointer = SqliteSaver(checkpointer_conn)
    return builder.compile(checkpointer=checkpointer)


def create_app_graph(database_path: str | Path = "data/vulkanmind.sqlite3") -> StateGraph:
    return build_graph(database_path)
