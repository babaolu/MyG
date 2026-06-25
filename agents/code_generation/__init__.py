from __future__ import annotations

from orchestrator.state import PlatformContext

from .agent import generate_vulkan_hpp_boilerplate
from .templates import cmake_for_platform
from .validator import validate_cpp_source


class CodeGenerationResult:
    """Lightweight result wrapper — this agent still produces C++ + CMake
    through the deterministic pipeline; the LLM remains available via
    `llm.LLMClient` for callers that want a structured refinement pass."""

    def __init__(
        self,
        generated_code: str,
        cmake_snippet: str,
        validation_passed: bool,
        validation_output: str,
        build_log: str,
    ) -> None:
        self.generated_code = generated_code
        self.cmake_snippet = cmake_snippet
        self.validation_passed = validation_passed
        self.validation_output = validation_output
        self.build_log = build_log


def validate_generated_code(code: str, cmake_snippet: str, context: PlatformContext) -> tuple[bool, str, str]:
    if context.thermal.mode == "THERMAL_THROTTLE":
        return False, "", "THERMAL_THROTTLE: compilation processes frozen"
    import tempfile
    from pathlib import Path

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
    from orchestrator.state import coerce_state

    state_model = coerce_state(state)
    context = state_model.platform_context
    if context is None:
        return {
            "error": "platform_context is required before code generation",
            "agent_trace": state_model.agent_trace,
        }
    generated_code = generate_vulkan_hpp_boilerplate(context)
    cmake_snippet = cmake_for_platform(context)
    validation_passed, validation_output, build_log = validate_generated_code(generated_code, cmake_snippet, context)
    return {
        "generated_code": generated_code,
        "cmake_snippet": cmake_snippet,
        "validation_passed": validation_passed,
        "validation_output": validation_output or build_log,
        "build_log": build_log,
        "agent_trace": state_model.agent_trace + ["code_generation_node generated Vulkan-Hpp/VMA code"],
    }
