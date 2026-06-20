from __future__ import annotations

from typing import Any

import psutil
from pydantic import BaseModel, Field

from .context_schema import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)
from .detector import (
    detect_host_platform as _detect_host_platform,
)
from .detector import (
    detect_target_platform as _detect_target_platform,
)
from .detector import (
    determine_cross_compile as _determine_cross_compile,
)


class PlatformIntelligenceResult(BaseModel):
    platform_context: PlatformContext
    trace: list[str] = Field(default_factory=list)


def _cpu_temp() -> float | None:
    temps = psutil.sensors_temperatures()
    for values in temps.values():
        for entry in values:
            current = float(getattr(entry, "current", 0.0))
            if current > 0:
                return current
    return None


def _adb_thermal_temp() -> float | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["adb", "shell", "for f in /sys/class/thermal/thermal_zone*/temp; do read t < $f; echo $t; done"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        try:
            value = float(line) / 1000.0
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return max(values) if values else None


def _battery_percent() -> float | None:
    battery = psutil.sensors_battery()
    return float(battery.percent) if battery and battery.percent is not None else None


def _derive_mode(cpu_load: float, temp: float | None, ram_available_percent: float, target_os: str) -> str:
    if temp is not None:
        threshold = 85.0 if target_os == "Android" else 92.0
        if temp > threshold:
            return "THERMAL_THROTTLE"
    if cpu_load < 40.0 and (temp is None or temp < 75.0) and ram_available_percent > 60.0:
        return "AGGRESSIVE"
    if cpu_load < 70.0:
        return "NORMAL"
    return "CONSERVATIVE"


def _worker_count(mode: str) -> int:
    cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
    return {
        "AGGRESSIVE": max(cpu_count - 1, 1),
        "NORMAL": max(cpu_count // 2, 1),
        "CONSERVATIVE": 1,
        "THERMAL_THROTTLE": 0,
    }[mode]


def read_thermal_state(target: TargetContext) -> ThermalContext:
    temp = _adb_thermal_temp() if target.os == "Android" else _cpu_temp()
    cpu_load = float(psutil.cpu_percent(interval=1))
    memory = psutil.virtual_memory()
    available_ram_percent = float(memory.available / memory.total * 100.0) if memory.total else 100.0
    mode = _derive_mode(cpu_load, temp, available_ram_percent, target.os)
    return ThermalContext(
        mode=mode,
        available_cores=max(psutil.cpu_count(logical=True) or 1, 1),
        cpu_load_percent=cpu_load,
        cpu_temp_celsius=temp,
        available_ram_percent=available_ram_percent,
        battery_percent=_battery_percent(),
        max_parallel_workers=_worker_count(mode),
    )


def detect_host_platform() -> HostContext:
    return _detect_host_platform()


def detect_target_platform(user_declared: dict[str, Any] | None, adb_connected: bool) -> TargetContext:
    return _detect_target_platform(user_declared, adb_connected)


def determine_cross_compile(host: HostContext, target: TargetContext) -> ToolchainContext:
    return _determine_cross_compile(host, target)


def build_platform_context(
    user_declared: dict[str, Any] | None = None,
    adb_connected: bool = False,
) -> PlatformContext:
    host = detect_host_platform()
    target = detect_target_platform(user_declared, adb_connected)
    toolchain = determine_cross_compile(host, target)
    thermal = read_thermal_state(target)
    runtime_validation_available = host.os == target.os and host.arch == target.arch
    return PlatformContext(
        host=host,
        target=target,
        toolchain=toolchain,
        thermal=thermal,
        runtime_validation_available=runtime_validation_available,
    )


def build_agent_system_prompt(platform_context: PlatformContext) -> str:
    serialized = platform_context.model_dump_json(indent=2)
    return (
        "You are a VulkanMind specialist agent. Every response must be structured data, not freeform prose. "
        "Generated C++ must use Vulkan-Hpp, RAII, std::expected or std::optional for errors, and VMA for device allocations. "
        "Active PlatformContext follows:\n"
        f"{serialized}"
    )


def platform_intelligence_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        user_declared = state.get("target_platform_declared") or state.get("target_platform")
        adb_connected = bool(state.get("adb_connected", False))
        context = build_platform_context(user_declared=user_declared, adb_connected=adb_connected)
        trace = state.get("agent_trace", []) + [
            f"platform_intelligence_node detected host={context.host.os}/{context.host.arch} target={context.target.source}"
        ]
        return {"platform_context": context, "agent_trace": trace}
    except Exception as exc:
        return {"error": str(exc), "agent_trace": state.get("agent_trace", []) + ["platform_intelligence_node failed"]}
