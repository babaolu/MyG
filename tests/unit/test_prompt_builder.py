"""Unit tests for ``llm.prompt_builder``.

The builder's job is to assemble the runtime system prompt from the immutable
AbsoluteRules block and an injected PlatformContext (plus optional improvement
memory). Tests pin both the rules and the structural so a future edit cannot
silently change agent contract.
"""
from __future__ import annotations

from llm.prompt_builder import ABSOLUTE_RULES, build_system_prompt, platform_context_summary
from orchestrator.state import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)


def _platform_context() -> PlatformContext:
    return PlatformContext(
        host=HostContext(os="Linux", arch="x86_64", compiler="clang++", compiler_version="18", auto_detected=True),
        target=TargetContext(
            os="Android",
            arch="arm64-v8a",
            gpu_vendor="Qualcomm",
            gpu_model="Adreno 740",
            vulkan_version="1.3",
            supported_extensions=["VK_KHR_timeline_semaphore"],
            unsupported_extensions=[],
            memory_heaps=[],
            quirk_profile={"prefer_tiling_optimal": True, "ubwc_compression": "enable"},
            auto_detected=False,
            source="user_declared",
        ),
        toolchain=ToolchainContext(is_cross_compile=True),
        thermal=ThermalContext(mode="NORMAL", available_cores=8, cpu_load_percent=10.0, available_ram_percent=70.0, max_parallel_workers=4),
        runtime_validation_available=False,
    )


def test_absolute_rules_include_required_invariants() -> None:
    """Absolute Rules must enumerate the immutable engineering contract."""
    assert "Vulkan-Hpp" in ABSOLUTE_RULES
    assert "RAII" in ABSOLUTE_RULES
    assert "std::expected" in ABSOLUTE_RULES
    assert "VMA" in ABSOLUTE_RULES
    assert "structured data" in ABSOLUTE_RULES


def test_build_system_prompt_includes_platform_context_json() -> None:
    prompt = build_system_prompt(_platform_context())
    assert ABSOLUTE_RULES in prompt
    assert "Active PlatformContext" in prompt
    assert "Qualcomm" in prompt
    assert "Adreno 740" in prompt


def test_build_system_prompt_with_improvement_context_appends_after_platform() -> None:
    memory = "TRUSTED SKILLS:\n- sample"
    prompt = build_system_prompt(_platform_context(), improvement_context=memory)
    platform_idx = prompt.find("Active PlatformContext")
    memory_idx = prompt.find("TRUSTED SKILLS")
    assert platform_idx >= 0
    assert memory_idx > platform_idx


def test_build_system_prompt_without_improvement_omits_appendix() -> None:
    prompt = build_system_prompt(_platform_context())
    assert "TRUSTED SKILLS" not in prompt
    assert "====" not in prompt or "Active PlatformContext" in prompt


def test_platform_context_summary_filters_to_routing_fields() -> None:
    summary = platform_context_summary(_platform_context())
    assert "Qualcomm" in summary
    assert "prefer_tiling_optimal" in summary
    assert "vulkan_version" in summary
    assert "compiler_version" not in summary  # noise removed for classifier prompts
