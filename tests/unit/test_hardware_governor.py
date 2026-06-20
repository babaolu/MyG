from agents.debugger.hardware_governor import can_compile, worker_limit
from orchestrator.state import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)


def _context(mode: str, workers: int) -> PlatformContext:
    return PlatformContext(
        host=HostContext(os="Linux", arch="x86_64", compiler="clang++", compiler_version="18", auto_detected=True),
        target=TargetContext(os="Linux", arch="x86_64", gpu_vendor="AMD", gpu_model="Radeon", vulkan_version="1.3", source="user_declared"),
        toolchain=ToolchainContext(is_cross_compile=False),
        thermal=ThermalContext(mode=mode, available_cores=8, cpu_load_percent=10.0, available_ram_percent=70.0, max_parallel_workers=workers),
        runtime_validation_available=True,
    )


def test_can_compile_false_for_thermal_throttle() -> None:
    assert can_compile(_context("THERMAL_THROTTLE", 0)) is False


def test_worker_limit_uses_governor() -> None:
    assert worker_limit(_context("NORMAL", 4)) == 4
