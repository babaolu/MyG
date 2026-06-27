from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import structlog

from agents.self_improvement.skill_extractor import SkillExtractor
from db.execution_traces import ExecutionTrace, ExecutionTraceStore
from orchestrator.state import (
    BugClassification,
    BugHypothesis,
    VulkanMindState,
    coerce_state,
    normalize_node_return,
)

# Pattern library is the actual write-back target. Earlier versions passed the
# skill_extractor instance to itself — bug. The module import below carries
# through every call to write_back_to_pattern_library.
from . import pattern_library as _pattern_library_module
from .classifier import classify_bug
from .pattern_library import match_patterns


class DebuggerResult:
    """Lightweight result container kept for backwards API compatibility."""

    def __init__(
        self,
        bug_classification: BugClassification,
        bug_hypotheses: list[BugHypothesis],
        active_fix: str | None = None,
        trace: list[str] | None = None,
    ) -> None:
        self.bug_classification = bug_classification
        self.bug_hypotheses = bug_hypotheses
        self.active_fix = active_fix
        self.trace = trace or []


def debugger_node(state: dict) -> dict:
    state_model = coerce_state(state)
    context = state_model.platform_context
    if context is None:
        return {
            "error": "platform_context is required before debugging",
            "agent_trace": state_model.agent_trace,
        }
    classification = classify_bug(
        state_model.validation_output,
        state_model.build_log,
        state_model.user_request,
    )
    text = "\n".join(
        part for part in [state_model.validation_output, state_model.build_log, state_model.user_request] if part
    )
    patterns = match_patterns(text, context.target.gpu_vendor)
    hypotheses = [hypothesis for pattern in patterns for hypothesis in pattern.hypotheses]
    active_fix = hypotheses[0].fix_template if hypotheses else None
    return normalize_node_return({
        "bug_classification": classification,
        "bug_hypotheses": hypotheses,
        "active_fix": active_fix,
        "agent_trace": state_model.agent_trace + ["debugger_node classified bug and selected hypotheses"],
    })


def post_session_record(
    state: VulkanMindState,
    trace_store: ExecutionTraceStore,
    skill_extractor: SkillExtractor,
) -> None:
    trace = _trace_from_state(state)
    trace_store.record(trace)
    skill_extracted = False
    skill_id = None
    if skill_extractor.should_extract(trace):
        skill = skill_extractor.extract(trace, state.platform_context)
        if skill is not None:
            skill_id = skill_extractor.save_skill(skill, trace)
            skill_extractor.write_back_to_pattern_library(skill, _pattern_library_module)
            skill_extracted = True
    structlog.get_logger("vulkanmind.self_improvement.execution_trace").info(
        "execution_trace_recorded",
        trace_id=trace.trace_id,
        outcome=trace.outcome,
        iterations_required=trace.iterations_required,
        skill_extracted=skill_extracted,
        skill_id=skill_id,
    )


def _trace_from_state(state: VulkanMindState) -> ExecutionTrace:
    context = state.platform_context
    winning_hypothesis = max(state.bug_hypotheses, key=lambda item: item.probability, default=None)
    pattern_matched = (
        state.bug_classification.patterns[0]
        if state.bug_classification and state.bug_classification.patterns
        else None
    )
    if pattern_matched is None and winning_hypothesis is not None:
        pattern_matched = _symptom_from_hypothesis(winning_hypothesis)
    validation_text = "\n".join(part for part in [state.validation_output, state.build_log] if part)
    outcome = "success" if state.validation_passed else "failure" if state.active_fix is None else "partial"
    iterations = sum(1 for entry in state.agent_trace if "debugger" in entry or "debug" in entry.lower())
    return ExecutionTrace(
        trace_id=state.trace_id or str(uuid4()),
        session_id=state.session_id,
        timestamp=datetime.now(UTC).isoformat(),
        task_type=state.task_type if state.task_type in {"code_generation", "debug", "knowledge_query"} else "debug",
        gpu_vendor=context.target.gpu_vendor if context else "unknown",
        gpu_model=context.target.gpu_model if context else "unknown",
        vulkan_version=context.target.vulkan_version if context else "unknown",
        target_os=context.target.os if context else "unknown",
        user_request=state.user_request,
        generated_code=state.generated_code,
        active_fix=state.active_fix,
        validation_passed=state.validation_passed,
        bug_classification=state.bug_classification.classification if state.bug_classification else None,
        bug_pattern_matched=pattern_matched,
        hypothesis_used=winning_hypothesis.description if winning_hypothesis else None,
        hypothesis_probability=winning_hypothesis.probability if winning_hypothesis else None,
        outcome=outcome,
        iterations_required=max(iterations, 1),
        validation_stderr=validation_text,
        clang_tidy_warnings=validation_text.lower().count("warning") if validation_text else 0,
        spirv_errors=validation_text.lower().count("spirv") if validation_text else 0,
    )


def _symptom_from_hypothesis(hypothesis: BugHypothesis) -> str:
    text = hypothesis.description.lower()
    if "swapchain" in text or "viewport" in text or "blend" in text:
        return "BLACK_SCREEN"
    if "shader" in text or "barrier" in text or "bounds" in text:
        return "GPU_HANG"
    if "layout" in text:
        return "VALIDATION_LAYOUT"
    if "stage" in text or "synchronization" in text:
        return "VALIDATION_SYNC"
    return "UNKNOWN"
