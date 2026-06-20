from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TypeVar
from uuid import uuid4

import structlog
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from agents.platform_intelligence.agent import build_agent_system_prompt
from agents.self_improvement.skill_extractor import SkillExtractor
from db.execution_traces import ExecutionTrace, ExecutionTraceStore
from orchestrator.state import BugClassification, BugHypothesis, PlatformContext, VulkanMindState

from .classifier import classify_bug
from .pattern_library import match_patterns


class DebuggerResult(BaseModel):
    bug_classification: BugClassification
    bug_hypotheses: list[BugHypothesis]
    active_fix: str | None = None
    trace: list[str]


T = TypeVar("T", bound=BaseModel)


def call_claude_structured(platform_context: PlatformContext, request: str, response_model: type[T], max_tokens: int = 2048) -> T:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Claude API calls")
    client = Anthropic(api_key=api_key)
    prompt = (
        "Return only valid JSON matching this Pydantic schema:\n"
        f"{json.dumps(response_model.model_json_schema(), indent=2)}\n"
        f"User request: {request}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=build_agent_system_prompt(platform_context),
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    try:
        return response_model.model_validate_json(text)
    except ValidationError as exc:
        return response_model.model_validate({"error": str(exc), "raw": text})


def debugger_node(state: dict) -> dict:
    context = state.get("platform_context")
    if context is None:
        return {"error": "platform_context is required before debugging", "agent_trace": state.get("agent_trace", [])}
    if isinstance(context, dict):
        context = PlatformContext.model_validate(context)
    classification = classify_bug(state.get("validation_output"), state.get("build_log"), state.get("user_request"))
    text = "\n".join(part for part in [state.get("validation_output"), state.get("build_log"), state.get("user_request")] if part)
    patterns = match_patterns(text, context.target.gpu_vendor)
    hypotheses = [hypothesis for pattern in patterns for hypothesis in pattern.hypotheses]
    active_fix = hypotheses[0].fix_template if hypotheses else None
    trace = state.get("agent_trace", []) + ["debugger_node classified bug and selected hypotheses"]
    return {
        "bug_classification": classification,
        "bug_hypotheses": hypotheses,
        "active_fix": active_fix,
        "agent_trace": trace,
    }


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
            skill_extractor.write_back_to_pattern_library(skill, skill_extractor)
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
    pattern_matched = state.bug_classification.patterns[0] if state.bug_classification and state.bug_classification.patterns else None
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
