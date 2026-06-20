from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from agents.platform_intelligence.agent import build_agent_system_prompt
from orchestrator.state import PlatformContext

from .agent import generate_vulkan_hpp_boilerplate
from .templates import cmake_for_platform
from .validator import validate_cpp_source


class CodeGenerationResult(BaseModel):
    generated_code: str
    cmake_snippet: str
    validation_passed: bool
    validation_output: str
    build_log: str


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


def validate_generated_code(code: str, cmake_snippet: str, context: PlatformContext) -> tuple[bool, str, str]:
    if context.thermal.mode == "THERMAL_THROTTLE":
        return False, "", "THERMAL_THROTTLE: compilation processes frozen"
    with tempfile.TemporaryDirectory(prefix="vulkanmind_codegen_") as temp_dir:
        root = Path(temp_dir)
        source_dir = root / "src"
        source_dir.mkdir()
        source_file = source_dir / "main.cpp"
        source_file.write_text(code, encoding="utf-8")
        (root / "CMakeLists.txt").write_text(cmake_snippet, encoding="utf-8")
        build_dir = root / "build"
        result = validate_cpp_source(source_file, build_dir, parallel_workers=context.thermal.max_parallel_workers)
        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        return result.passed, output, "; ".join(result.faults)


def code_generation_node(state: dict) -> dict:
    context = state.get("platform_context")
    if context is None:
        return {"error": "platform_context is required before code generation", "agent_trace": state.get("agent_trace", [])}
    if isinstance(context, dict):
        context = PlatformContext.model_validate(context)
    generated_code = generate_vulkan_hpp_boilerplate(context)
    cmake_snippet = cmake_for_platform(context)
    validation_passed, validation_output, build_log = validate_generated_code(generated_code, cmake_snippet, context)
    trace = state.get("agent_trace", []) + ["code_generation_node generated Vulkan-Hpp/VMA code"]
    return {
        "generated_code": generated_code,
        "cmake_snippet": cmake_snippet,
        "validation_passed": validation_passed,
        "validation_output": validation_output or build_log,
        "build_log": build_log,
        "agent_trace": trace,
    }
