from __future__ import annotations

from pydantic import BaseModel, Field

from agents.self_update.changelog import ChangelogStore
from agents.self_update.monitors.khronos_monitor import KhronosMonitor, UpdateDiff
from agents.self_update.monitors.research_monitor import poll_research_sources
from agents.self_update.monitors.vendor_monitor import poll_vendor_blogs
from db.build_queue_store import BuildQueueStore
from orchestrator.state import PlatformContext


class SelfUpdateResult(BaseModel):
    pending_updates: list[UpdateDiff] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


def self_update_node(state: dict) -> dict:
    context = state.get("platform_context")
    if context is None:
        return {"error": "platform_context is required before self update", "agent_trace": state.get("agent_trace", [])}
    if isinstance(context, dict):
        context = PlatformContext.model_validate(context)
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
    trace = state.get("agent_trace", []) + ["self_update_node polled monitors without applying unconfirmed changes"]
    return {
        "pending_updates": store.pending_updates(),
        "agent_trace": trace,
    }
