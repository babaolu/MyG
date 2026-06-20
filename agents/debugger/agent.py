from __future__ import annotations

import json
import os
from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from agents.platform_intelligence.agent import build_agent_system_prompt
from orchestrator.state import BugClassification, BugHypothesis, PlatformContext

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
