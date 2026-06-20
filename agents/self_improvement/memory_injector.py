from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from agents.debugger.pattern_library import get_all_symptoms
from db.execution_traces import ExecutionTrace, ExecutionTraceStore
from db.skill_writebacks import SkillStore, VulkanSkill
from orchestrator.state import PlatformContext, VulkanMindState


class SessionMemory(BaseModel):
    relevant_skills: list[VulkanSkill]
    recent_fixes: list[ExecutionTrace]
    platform_hit_rate: dict[str, float]
    injected_at: str


class MemoryInjector:
    def __init__(
        self,
        skill_store: SkillStore,
        trace_store: ExecutionTraceStore,
    ):
        self.skill_store = skill_store
        self.trace_store = trace_store

    def build_session_memory(
        self,
        platform_context: PlatformContext,
        task_type: str,
    ) -> SessionMemory:
        _ = task_type
        skills = self.skill_store.get_trusted_skills()
        platform_skills = [
            skill for skill in skills
            if skill.gpu_vendor is None or skill.gpu_vendor == platform_context.target.gpu_vendor
        ]
        recent = self.trace_store.get_by_platform(
            gpu_vendor=platform_context.target.gpu_vendor,
            vulkan_version=platform_context.target.vulkan_version,
            outcome="success",
            limit=10,
        )
        hit_rates: dict[str, float] = {}
        for symptom in get_all_symptoms():
            rate = self.trace_store.get_pattern_hit_rate(symptom, platform_context.target.gpu_vendor)
            if rate > 0:
                hit_rates[symptom] = rate
        return SessionMemory(
            relevant_skills=platform_skills,
            recent_fixes=recent,
            platform_hit_rate=hit_rates,
            injected_at=datetime.now(UTC).isoformat(),
        )

    def format_for_system_prompt(
        self,
        memory: SessionMemory,
        max_tokens: int = 2000,
    ) -> str:
        blocks = ["=== VulkanMind Accumulated Knowledge ===", ""]
        blocks.append(f"TRUSTED SKILLS ({len(memory.relevant_skills)} available for this platform):")
        skill_blocks = [_format_skill(skill) for skill in memory.relevant_skills]
        blocks.extend(_truncate_blocks(skill_blocks, max_tokens // 3))

        blocks.append("")
        blocks.append(f"RECENT SUCCESSFUL FIXES (same platform, last {len(memory.recent_fixes)} sessions):")
        fix_blocks = [_format_trace(trace) for trace in memory.recent_fixes]
        blocks.extend(_truncate_blocks(fix_blocks, max_tokens // 3))

        blocks.append("")
        blocks.append("PLATFORM PATTERN HIT RATES:")
        rate_blocks = [f"- {symptom}: {rate:.0%} success rate" for symptom, rate in memory.platform_hit_rate.items()]
        blocks.extend(_truncate_blocks(rate_blocks, max_tokens // 3))
        return "\n".join(block for block in blocks if block)

    def inject_into_state(
        self,
        state: VulkanMindState,
        memory: SessionMemory,
    ) -> VulkanMindState:
        formatted = self.format_for_system_prompt(memory)
        state.session_memory = memory
        state.improvement_context = formatted
        return state


def _format_skill(skill: VulkanSkill) -> str:
    procedure = skill.fix_procedure.replace("\n", " ")[:180]
    return f"- {skill.name} | {skill.symptom} | successes={skill.times_successful} | {procedure}"


def _format_trace(trace: ExecutionTrace) -> str:
    fix = (trace.active_fix or "no active fix").replace("\n", " ")[:160]
    return f"- {trace.trace_id} | {trace.bug_pattern_matched or trace.bug_classification or 'unknown'} | fix={fix} | iterations={trace.iterations_required}"


def _truncate_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for block in blocks:
        estimated = max(1, len(block.split()) // 4)
        if used + estimated > max_tokens:
            break
        selected.append(block)
        used += estimated
    return selected
