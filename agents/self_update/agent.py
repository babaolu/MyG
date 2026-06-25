from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from agents.self_improvement.pattern_curator import PatternCurator
from agents.self_improvement.prompt_refiner import PromptRefiner
from agents.self_update.changelog import ChangelogStore
from agents.self_update.monitors.khronos_monitor import KhronosMonitor, UpdateDiff
from agents.self_update.monitors.research_monitor import poll_research_sources
from agents.self_update.monitors.vendor_monitor import poll_vendor_blogs
from db.build_queue_store import BuildQueueStore
from orchestrator.state import coerce_state


class SelfUpdateResult(BaseModel):
    pending_updates: list[UpdateDiff] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


def self_update_node(state: dict) -> dict:
    state_model = coerce_state(state)
    context = state_model.platform_context
    if context is None:
        return {
            "error": "platform_context is required before self update",
            "agent_trace": state_model.agent_trace,
        }
    queue_store = state.get("build_queue_store") or BuildQueueStore()
    store = ChangelogStore(store=queue_store)
    current_version = state.get("current_spec_version", "1.3.296")
    diffs: list[UpdateDiff] = []
    khronos_diff = KhronosMonitor(current_spec_version=current_version).poll()
    if khronos_diff:
        diffs.append(khronos_diff)
        store.save(khronos_diff)
    diffs.extend(poll_vendor_blogs())
    diffs.extend(poll_research_sources())
    return {
        "pending_updates": store.pending_updates(),
        "agent_trace": state_model.agent_trace + ["self_update_node polled monitors without applying unconfirmed changes"],
    }


def run_pattern_audit(
    curator: PatternCurator,
    refiner: PromptRefiner,
) -> None:
    logger = structlog.get_logger("vulkanmind.self_update.pattern_audit")
    report = curator.run_weekly_audit()
    proposals = refiner.analyse_failure_patterns(lookback_days=30)
    refiner.save_proposals(proposals)
    review_items = list(report.requires_human_review) + [proposal.proposal_id for proposal in proposals]
    if review_items:
        logger.warning(
            "self_improvement_human_review_required",
            review_items=review_items,
            proposals=len(proposals),
        )
    logger.info(
        "pattern_audit_completed",
        report=report.model_dump(),
        proposals=len(proposals),
    )
