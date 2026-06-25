"""Runtime system-prompt builder for VulkanMind LLM agents.

The prompt is split into an Absolute Rules block (immutable engineering
contract) and a dynamic block injected from the per-session PlatformContext
+ accumulated self-improvement memory. The platform block is the only thing
that varies between providers/models; the Absolute Rules never change, so we
keep them as a separate constant for auditing.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.state import PlatformContext

"""Absolute Rules every VulkanMind agent must follow.

These rules are passed verbatim into the system prompt. Editing them is a
breaking change to agent behaviour; changes require prompt-refinement
proposals through `agents.self_improvement.prompt_refiner.PromptRefiner`.
"""
ABSOLUTE_RULES: str = """\
You are a VulkanMind specialist agent focused on modern Vulkan / Vulkan-Hpp /
VMA graphics engineering. Every response must be structured data, never freeform prose.

Absolute Rules:
1. Generated C++ must use Vulkan-Hpp (the C++ binding) — never raw C vulkan.h.
2. All resources follow RAII: every `vk::*` handle has a single owning scope and
   a deterministic deleter. No raw `Vk*` lifetimes on the stack.
3. Errors are reported through `std::expected<T, Error>` or `std::optional<T>`.
   No C-style return codes on the success path. No `throw` for recoverable errors.
4. Device allocations use VMA (Vulkan Memory Allocator). The agent must show
   the `VmaAllocationCreateInfo` and `VmaAllocator` interaction explicitly.
5. Shader / pipeline objects follow C++ Core Guidelines: const correctness,
   rule-of-five, no naked `new`/`delete`, no `using namespace` in headers.
6. Cross-platform truth: the agent must respect quirks from the injected
   PlatformContext (e.g. Qualcomm UBWC, ARM TBDR). Where the quirk contradicts
   a Vulkan reference page, the quirk wins for code generation.
7. The agent returns ONLY valid JSON matching the requested schema. No
   explanations, no markdown fences, no trailing prose.
"""


def build_system_prompt(
    platform_context: PlatformContext,
    *,
    improvement_context: str | None = None,
    platform_context_block: str | None = None,
) -> str:
    """Assemble the runtime system prompt.

    Args:
        platform_context: The active VulkanMind PlatformContext (Pydantic).
        improvement_context: Optional formatted memory block from
            `agents.self_improvement.memory_injector.MemoryInjector.format_for_system_prompt`.
        platform_context_block: Optional pre-formatted PlatformContext string.
            When ``None`` (typical) we `model_dump_json` for the agent.
    """
    platform_block = platform_context_block or platform_context.model_dump_json(indent=2)
    parts: list[str] = [ABSOLUTE_RULES, "", "Active PlatformContext:", platform_block]
    if improvement_context is not None:
        parts.extend(["", improvement_context])
    return "\n".join(parts)


def platform_context_summary(platform_context: PlatformContext) -> str:
    """Compact one-line context summary for classifier prompts (e.g. router).

    Long PlatformContext dumps bloat small classifier prompts; this helper keeps
    only the fields that change routing decisions.
    """
    target = platform_context.target
    summary = {
        "os": target.os,
        "gpu_vendor": target.gpu_vendor,
        "gpu_model": target.gpu_model,
        "vulkan_version": target.vulkan_version,
        "quirk_keys": sorted(target.quirk_profile.keys()),
        "thermal_mode": platform_context.thermal.mode,
    }
    return json.dumps(summary, indent=2)
