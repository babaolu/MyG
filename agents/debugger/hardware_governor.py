from __future__ import annotations

from orchestrator.state import PlatformContext


def current_mode(context: PlatformContext) -> str:
    return context.thermal.mode


def can_compile(context: PlatformContext) -> bool:
    return context.thermal.mode != "THERMAL_THROTTLE" and context.thermal.max_parallel_workers > 0


def worker_limit(context: PlatformContext) -> int:
    return context.thermal.max_parallel_workers
