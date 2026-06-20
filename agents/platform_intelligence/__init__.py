from __future__ import annotations

from .agent import build_agent_system_prompt, build_platform_context, platform_intelligence_node
from .context_schema import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)
from .detector import (
    PlatformContextError,
    detect_host_platform,
    detect_target_platform,
    determine_cross_compile,
)

__all__ = [
    "HostContext",
    "PlatformContext",
    "PlatformContextError",
    "TargetContext",
    "ThermalContext",
    "ToolchainContext",
    "build_agent_system_prompt",
    "build_platform_context",
    "detect_host_platform",
    "detect_target_platform",
    "determine_cross_compile",
    "platform_intelligence_node",
]
