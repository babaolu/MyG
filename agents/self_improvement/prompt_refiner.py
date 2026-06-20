from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog
from anthropic import Anthropic
from pydantic import BaseModel

from db.execution_traces import ExecutionTraceStore
from db.skill_writebacks import SkillStore


class PromptRefinementProposal(BaseModel):
    proposal_id: str
    created_at: str
    agent_target: str
    current_prompt_excerpt: str
    proposed_change: str
    rationale: str
    evidence: list[str]
    status: str = "pending"


class _ProposalResponse(BaseModel):
    proposals: list[PromptRefinementProposal]
    confidence: float


class PromptRefiner:
    def __init__(
        self,
        anthropic_client: Anthropic,
        trace_store: ExecutionTraceStore,
        skill_store: SkillStore | None = None,
    ):
        self.anthropic_client = anthropic_client
        self.trace_store = trace_store
        self.skill_store = skill_store

    def analyse_failure_patterns(
        self,
        lookback_days: int = 30,
    ) -> list[PromptRefinementProposal]:
        if self.anthropic_client is None:
            return []
        since = datetime.now(UTC) - timedelta(days=lookback_days)
        failures = self.trace_store.get_failures_since(since)
        grouped: dict[str, list[Any]] = {}
        for trace in failures:
            grouped.setdefault(_agent_target(trace.task_type), []).append(trace)
        proposals: list[PromptRefinementProposal] = []
        for agent_target, traces in grouped.items():
            if len(traces) < 5:
                continue
            response = self._ask_claude(agent_target, traces)
            if response.confidence >= 0.7:
                proposals.extend(response.proposals)
        return proposals

    def save_proposals(
        self,
        proposals: list[PromptRefinementProposal],
    ) -> None:
        if self.skill_store is None:
            return
        for proposal in proposals:
            self.skill_store.save_prompt_proposal(proposal)

    def get_pending_proposals(self) -> list[PromptRefinementProposal]:
        if self.skill_store is None:
            return []
        return list(self.skill_store.get_prompt_proposals("pending"))

    def apply_approved_proposal(
        self,
        proposal_id: str,
    ) -> None:
        if self.skill_store is None:
            return
        proposals = self.skill_store.get_prompt_proposals()
        proposal = next((item for item in proposals if item.proposal_id == proposal_id), None)
        if proposal is None or proposal.status != "approved":
            raise ValueError("proposal must be approved before application")
        self.skill_store.apply_prompt_proposal(proposal)
        structlog.get_logger("vulkanmind.self_improvement.prompt_refiner").info(
            "prompt_refinement_applied",
            proposal_id=proposal_id,
            agent_target=proposal.agent_target,
        )

    def approve_proposal(self, proposal_id: str, confirmed: bool) -> bool:
        if self.skill_store is None:
            return False
        if confirmed:
            self.skill_store.approve_prompt_proposal(proposal_id)
        else:
            self.skill_store.reject_prompt_proposal(proposal_id)
        return True

    def get_runtime_prompt_override(self, agent_target: str) -> str | None:
        if self.skill_store is None:
            return None
        return self.skill_store.get_prompt_override(agent_target)

    def _ask_claude(self, agent_target: str, traces) -> _ProposalResponse:
        prompt = json.dumps(
            {
                "agent_target": agent_target,
                "failure_traces": [trace.model_dump() for trace in traces],
                "schema": _ProposalResponse.model_json_schema(),
            },
            indent=2,
        )
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=(
                    "You are refining VulkanMind system prompts from execution failures. "
                    "Return only structured JSON. Propose precise prompt changes only when confidence is high."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            parsed = _ProposalResponse.model_validate_json(text)
            for proposal in parsed.proposals:
                proposal.proposal_id = f"prompt-{uuid4()}"
                proposal.created_at = datetime.now(UTC).isoformat()
                proposal.status = "pending"
            return parsed
        except Exception:
            return _ProposalResponse(proposals=[], confidence=0.0)


def _agent_target(task_type: str) -> str:
    if task_type == "debug":
        return "debugger"
    if task_type == "knowledge_query":
        return "knowledge_retrieval"
    return "code_generation"
