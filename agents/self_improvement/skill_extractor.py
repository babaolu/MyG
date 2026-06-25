from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from db.execution_traces import ExecutionTrace
from db.skill_writebacks import SkillStore, VulkanSkill
from orchestrator.state import (
    BugHypothesis,
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)


class ExtractedSkill(BaseModel):
    name: str
    symptom: str
    domain: str
    fix_procedure: str
    code_template: str | None = None
    cmake_snippet: str | None = None
    validation_step: str
    is_platform_specific: bool
    confidence_rationale: str


class SkillExtractor:
    def __init__(self, llm_client, skill_store: SkillStore):
        self.llm_client = llm_client
        self.skill_store = skill_store

    def should_extract(self, trace: ExecutionTrace) -> bool:
        return (
            trace.outcome == "success"
            and trace.iterations_required >= 2
            and bool(trace.bug_pattern_matched)
        )

    def extract(
        self,
        trace: ExecutionTrace,
        platform_context: PlatformContext,
    ) -> VulkanSkill | None:
        if self.llm_client is None:
            return None
        from llm.client import LLMMessage

        prompt = self._build_extraction_prompt(trace, platform_context)
        try:
            response = self.llm_client.complete_structured(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are extracting a reusable Vulkan graphics skill from a successful debug trace. "
                            "The skill must be expressed as a platform-aware, step-by-step procedure that another "
                            "agent could follow to resolve the same class of problem without re-deriving the solution. "
                            "The fix_procedure must be precise and reference specific Vulkan-Hpp APIs and VMA interfaces "
                            "where applicable. The validation_step must be a concrete, executable check."
                        ),
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                ExtractedSkill,
                max_tokens=2048,
            )
            extracted = response
        except Exception:
            return None
        return self._to_skill(extracted, trace, platform_context)

    def save_skill(
        self,
        skill: VulkanSkill,
        trace: ExecutionTrace,
    ) -> str:
        self.skill_store.save(skill)
        return skill.skill_id

    def write_back_to_pattern_library(
        self,
        skill: VulkanSkill,
        pattern_library: Any,
    ) -> None:
        """Write the extracted skill back into the pattern library.

        ``pattern_library`` is the actual `agents.debugger.pattern_library`
        module; some legacy callers may still pass the ``SkillExtractor``
        instance — we treat that as a no-op instead of corrupting state by
        writing into ourselves.
        """
        write_back = getattr(pattern_library, "write_back_fix", None)
        if write_back is None:
            # Self or unknown object — silently no-op. This used to silently
            # double-write into the pattern library, masking the real bug.
            return
        hypothesis = BugHypothesis(
            description=skill.fix_procedure,
            probability=0.9 if skill.confidence == "trusted" else 0.75,
            fix_template=skill.fix_procedure,
            validation_step=skill.validation_step,
            affected_platforms=[skill.gpu_vendor] if skill.gpu_vendor else None,
        )
        platform_context = _platform_from_skill(skill)
        write_back(
            symptom=skill.symptom,
            hypothesis=hypothesis,
            platform_context=platform_context,
            trace_id="",
            skill_id=skill.skill_id,
        )

    def _build_extraction_prompt(self, trace: ExecutionTrace, platform_context: PlatformContext) -> str:
        return json.dumps(
            {
                "trace": trace.model_dump(),
                "platform_context": platform_context.model_dump(),
                "schema": ExtractedSkill.model_json_schema(),
            },
            indent=2,
        )

    def _to_skill(
        self,
        extracted: ExtractedSkill,
        trace: ExecutionTrace,
        platform_context: PlatformContext,
    ) -> VulkanSkill:
        confidence = "high" if trace.iterations_required >= 3 else "medium"
        return VulkanSkill(
            skill_id=f"skill-{uuid4()}",
            created_at=datetime.now(UTC).isoformat(),
            name=extracted.name,
            symptom=extracted.symptom or trace.bug_pattern_matched or "unknown",
            domain=extracted.domain,
            gpu_vendor=platform_context.target.gpu_vendor if extracted.is_platform_specific else None,
            vulkan_version_min=platform_context.target.vulkan_version,
            target_os=platform_context.target.os if extracted.is_platform_specific else None,
            fix_procedure=extracted.fix_procedure,
            code_template=extracted.code_template,
            cmake_snippet=extracted.cmake_snippet,
            validation_step=extracted.validation_step,
            confidence=confidence,
        )


def _platform_from_skill(skill: VulkanSkill) -> PlatformContext:
    return PlatformContext(
        host=HostContext(os="unknown", arch="unknown", compiler="unknown", compiler_version="unknown"),
        target=TargetContext(
            os=skill.target_os or "unknown",
            arch="unknown",
            gpu_vendor=skill.gpu_vendor or "unknown",
            gpu_model="unknown",
            vulkan_version=skill.vulkan_version_min,
            source="user_declared",
        ),
        toolchain=ToolchainContext(is_cross_compile=False),
        thermal=ThermalContext(
            mode="NORMAL",
            available_cores=1,
            cpu_load_percent=0.0,
            available_ram_percent=100.0,
            max_parallel_workers=1,
        ),
        runtime_validation_available=False,
    )
