from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import agents.debugger.pattern_library as pattern_library
from agents.self_improvement.memory_injector import MemoryInjector
from agents.self_improvement.pattern_curator import PatternCurator
from agents.self_improvement.prompt_refiner import PromptRefiner
from agents.self_improvement.skill_extractor import SkillExtractor
from db.bug_history import BugHistoryStore
from db.build_queue_store import BuildQueueStore
from db.execution_traces import ExecutionTraceStore
from db.session_store import SessionStore
from db.skill_writebacks import SkillStore
from orchestrator.graph import build_graph, configure_self_improvement

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


CONFIG_PATH = Path(os.environ.get("VULKANMIND_CONFIG", "config.yaml"))


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


config = load_config()
storage_path = config.get("storage", {}).get("database_path", "data/vulkanmind.sqlite3")
session_store = SessionStore(storage_path)
bug_history_store = BugHistoryStore(storage_path)
build_queue_store = BuildQueueStore(storage_path)
trace_store = ExecutionTraceStore(storage_path)
skill_store = SkillStore(storage_path)
memory_injector = MemoryInjector(skill_store, trace_store)
skill_extractor = SkillExtractor(_anthropic_client(), skill_store)
prompt_refiner = PromptRefiner(_anthropic_client(), trace_store, skill_store)
pattern_curator = PatternCurator(skill_store, trace_store, pattern_library)
configure_self_improvement(
    storage_path,
    trace_store=trace_store,
    skill_store=skill_store,
    memory_injector=memory_injector,
    skill_extractor=skill_extractor,
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
        "attached_files": request.attached_files if hasattr(request, "attached_files") else [],
        "validation_output": request.validation_output,
        "build_log": request.build_log,
        "target_platform_declared": target_declared,
        "agent_trace": [],
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


def _qdrant_connected() -> bool:
    qdrant_config = config.get("qdrant", {})
    url = f"http://{qdrant_config.get('host', 'localhost')}:{qdrant_config.get('port', 6333)}/readyz"
    try:
        response = httpx.get(url, timeout=3)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
