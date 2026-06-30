from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load .env from the project root BEFORE any VulkanMind module reads
# ``os.environ`` for API keys (LLM factory, embedding client, etc.).
# Loading relative to ``__file__`` rather than ``cwd`` keeps the lookup
# deterministic regardless of where uvicorn was started from. ``override=False``
# makes the loader additive only — values already exported in the real shell
# environment win, since that's the standard 12-factor precedence.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# TEMP diagnostic: confirm the loader actually populated the key in this process.
import os as _os  # noqa: E402

print(f"[vm boot] NVIDIA_API_KEY={'present' if _os.environ.get('NVIDIA_API_KEY') else 'absent'} len={len(_os.environ.get('NVIDIA_API_KEY', ''))}", flush=True)  # noqa: E402

# First-party imports follow the env loader by design — the LLM factory,
# embedding client, etc. all read ``os.environ`` at module-import-time of their
# own modules, so we must populate the env before they are imported.
import agents.debugger.pattern_library as pattern_library  # noqa: E402
from agents.self_improvement.memory_injector import MemoryInjector  # noqa: E402
from agents.self_improvement.pattern_curator import PatternCurator  # noqa: E402
from agents.self_improvement.prompt_refiner import PromptRefiner  # noqa: E402
from agents.self_improvement.skill_extractor import SkillExtractor  # noqa: E402
from db.bug_history import BugHistoryStore  # noqa: E402
from db.build_queue_store import BuildQueueStore  # noqa: E402
from db.execution_traces import ExecutionTraceStore  # noqa: E402
from db.session_store import SessionStore  # noqa: E402
from db.skill_writebacks import SkillStore  # noqa: E402
from llm.factory import create_llm_client  # noqa: E402
from orchestrator.graph import build_graph, configure_self_improvement  # noqa: E402
from utils.config import CONFIG_PATH, load_config  # noqa: E402

logger = structlog.get_logger("vulkanmind")


class SessionStartRequest(BaseModel):
    user_request: str
    target_platform_declared: dict[str, Any] | None = None
    attached_files: list[str] = Field(default_factory=list)


class SessionMessageRequest(BaseModel):
    message: str
    validation_output: str | None = None
    build_log: str | None = None
    attached_files: list[str] = Field(default_factory=list)


class UpdateConfirmationRequest(BaseModel):
    update_id: str
    confirmed: bool


class SkillRetireRequest(BaseModel):
    reason: str


class ProposalApprovalRequest(BaseModel):
    confirmed: bool


class MessageResponse(BaseModel):
    agent_response: dict[str, Any]
    generated_code: str | None = None
    debug_report: dict[str, Any] | None = None
    knowledge_citations: list[dict[str, Any]] = Field(default_factory=list)


# ``load_config`` now lives in ``utils.config``; re-export it under the legacy
# ``main.load_config`` symbol so any external importer keeps working.
__all__ = ["CONFIG_PATH", "load_config"]


config = load_config()
storage_path = config.get("storage", {}).get("database_path", "data/vulkanmind.sqlite3")
session_store = SessionStore(storage_path)
bug_history_store = BugHistoryStore(storage_path)
build_queue_store = BuildQueueStore(storage_path)
trace_store = ExecutionTraceStore(storage_path)
skill_store = SkillStore(storage_path)

llm_client = create_llm_client(config.get("llm"))
print(f"[vm boot] llm_client after create_llm_client: {type(llm_client).__name__ if llm_client else 'None'}", flush=True)  # noqa: E402
memory_injector = MemoryInjector(skill_store, trace_store)
skill_extractor = SkillExtractor(llm_client, skill_store)
prompt_refiner = PromptRefiner(llm_client, trace_store, skill_store)
pattern_curator = PatternCurator(skill_store, trace_store, pattern_library)


def _graphify_reader():
    from tools.graphify_reader import build_graphify_reader

    return build_graphify_reader(config.get("graphify"))


configure_self_improvement(
    storage_path,
    trace_store=trace_store,
    skill_store=skill_store,
    memory_injector=memory_injector,
    skill_extractor=skill_extractor,
    llm_client=llm_client,
    graphify_reader=_graphify_reader(),
)
graph = build_graph(storage_path)
app = FastAPI(title="VulkanMind", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    qdrant_connected = _qdrant_connected()
    return {
        "status": "ok",
        "qdrant_connected": qdrant_connected,
        "hardware_governor_mode": config.get("hardware_governor", {}).get("default_mode", "NORMAL"),
    }


@app.post("/session/start")
def start_session(request: SessionStartRequest) -> dict[str, str]:
    session_id = session_store.create_session(request.user_request, request.target_platform_declared)
    return {"session_id": session_id}


@app.post("/session/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, request: SessionMessageRequest) -> MessageResponse:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    target_declared = session.get("target_platform_json") or {}
    if isinstance(target_declared, str):
        try:
            target_declared = json.loads(target_declared)
        except json.JSONDecodeError:
            target_declared = {}
    initial_state = {
        "session_id": session_id,
        "task_type": "unknown",
        "user_request": request.message,
        "attached_files": getattr(request, "attached_files", []),
        "validation_output": request.validation_output,
        "build_log": request.build_log,
        "target_platform_declared": target_declared,
        "agent_trace": [],
        # Reset the per-turn self-improvement loop. These fields are persisted
        # by the SqliteSaver checkpointer per ``thread_id``, so without an
        # explicit reset the second message of a session would inherit
        # ``self_improvement_phase="end"`` and ``session_memory!=None`` from
        # the previous run — which causes ``self_improvement_node`` to skip
        # ``router_node`` entirely and short-circuit straight to END.
        "self_improvement_phase": "start",
        "session_memory": None,
        "improvement_context": None,
        # Clear any stale error from previous turns
        "error": None,
        # NOTE: ``memory_injector`` / ``llm_client`` / ``graphify_reader`` are
        # NOT carried in state — they are runtime collaborators kept in
        # module-scoped singletons (``graph._MEMORY_INJECTOR`` etc.). Pushing
        # them into state would put unserialisable objects into the
        # langgraph checkpoint persistence path.
    }
    result = graph.invoke(initial_state, config={"configurable": {"thread_id": session_id}})
    result = dict(result) if not isinstance(result, dict) else result
    if result.get("bug_classification") is not None:
        classification = result["bug_classification"]
        bug_history_store.record(
            session_id=session_id,
            classification=classification.get("classification", "unknown"),
            hypotheses=result.get("bug_hypotheses", []),
            active_fix=result.get("active_fix"),
        )
    return MessageResponse(
        agent_response=result,
        generated_code=result.get("generated_code"),
        debug_report={
            "classification": result.get("bug_classification"),
            "hypotheses": result.get("bug_hypotheses", []),
            "active_fix": result.get("active_fix"),
        } if result.get("bug_classification") else None,
        knowledge_citations=result.get("retrieved_knowledge", []),
    )


@app.get("/session/{session_id}/platform_context")
def platform_context(session_id: str) -> dict[str, Any]:
    result = graph.invoke(
        {
            "session_id": session_id,
            "task_type": "platform_detect",
            "user_request": "return current platform context",
            "agent_trace": [],
        },
        config={"configurable": {"thread_id": session_id}},
    )
    result = dict(result) if not isinstance(result, dict) else result
    if result.get("platform_context") is None:
        raise HTTPException(status_code=404, detail="platform context unavailable")
    return result["platform_context"]


@app.get("/session/{session_id}/build_queue")
def build_queue(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "build_queue": build_queue_store.list_queue()}


@app.post("/session/{session_id}/confirm_update")
def confirm_update(session_id: str, request: UpdateConfirmationRequest) -> dict[str, Any]:
    try:
        diff = build_queue_store.confirm_update_diff(request.update_id, request.confirmed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="update not found") from exc
    return {"session_id": session_id, "update": diff.model_dump()}


@app.get("/skills")
def skills(
    gpu_vendor: str | None = None,
    domain: str | None = None,
    confidence: str | None = None,
) -> dict[str, Any]:
    return {
        "skills": [
            skill.model_dump()
            for skill in skill_store.get_all_active(
                gpu_vendor=gpu_vendor,
                domain=domain,
                confidence=confidence,
            )
        ]
    }


@app.get("/skills/trusted")
def trusted_skills() -> dict[str, Any]:
    return {"skills": [skill.model_dump() for skill in skill_store.get_trusted_skills()]}


@app.get("/skills/review")
def skills_review() -> dict[str, Any]:
    return {
        "skills_under_review": [
            skill.model_dump()
            for skill in skill_store.get_all_active()
            if skill.status == "under_review"
        ],
        "pending_proposals": [
            proposal.model_dump()
            for proposal in prompt_refiner.get_pending_proposals()
        ],
        "human_review": skill_store.get_human_review(),
    }


@app.post("/skills/{skill_id}/retire")
def retire_skill(skill_id: str, request: SkillRetireRequest) -> dict[str, Any]:
    skill_store.retire(skill_id, request.reason)
    skill_store.save_human_review(skill_id, "skill", request.reason)
    return {"success": True}


@app.post("/skills/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: str, request: ProposalApprovalRequest) -> dict[str, Any]:
    if not prompt_refiner.approve_proposal(proposal_id, request.confirmed):
        raise HTTPException(status_code=404, detail="proposal not found")
    applied = False
    if request.confirmed:
        try:
            prompt_refiner.apply_approved_proposal(proposal_id)
            applied = True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "applied": applied}


@app.get("/skills/stats")
def skills_stats() -> dict[str, Any]:
    return {
        "total_skills": len(skill_store.get_all_active()),
        "trusted_skills": len(skill_store.get_trusted_skills()),
        "total_traces": trace_store.total_traces(),
        "success_rate_by_platform": trace_store.success_rate_by_platform(),
        "top_symptoms_resolved": trace_store.top_symptoms_resolved(),
        "avg_iterations_to_resolve": trace_store.average_iterations_to_resolve(),
    }


class GraphPathRequest(BaseModel):
    path: str


@app.post("/graph/path")
def set_graph_path(request: GraphPathRequest) -> dict[str, Any]:
    from orchestrator.graph import set_graphify_path

    if set_graphify_path(request.path):
        return {"success": True, "path": request.path}
    raise HTTPException(status_code=404, detail="graph.json not found at that path")


def _qdrant_connected() -> bool:
    qdrant_config = config.get("qdrant", {})
    url = f"http://{qdrant_config.get('host', 'localhost')}:{qdrant_config.get('port', 6333)}/readyz"
    try:
        response = httpx.get(url, timeout=3)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
