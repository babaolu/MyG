from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from typing import Any

from .context_schema import HostContext, TargetContext, ToolchainContext
from .quirk_profiles import get_quirk_profile


class PlatformContextError(RuntimeError):
    pass


def _run_text(command: list[str], timeout: int = 10) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _compiler_info() -> tuple[str, str]:
    candidates = [
        ["clang++", "--version"],
        ["c++", "--version"],
        ["g++", "--version"],
        ["gcc", "--version"],
        ["cl", "/?"]
    ]
    for command in candidates:
        output = _run_text(command)
        if not output:
            continue
        first_line = output.splitlines()[0]
        if command[0] == "cl":
            return "cl", first_line
        return command[0], first_line
    return "unknown", "unknown"


def detect_host_platform() -> HostContext:
    compiler, version = _compiler_info()
    return HostContext(
        os=platform.system(),
        arch=platform.machine() or "unknown",
        compiler=compiler,
        compiler_version=version,
        auto_detected=True,
    )


def _adb_shell(command: str) -> str:
    return _run_text(["adb", "shell", command], timeout=15)


def _parse_vkjson(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def _extract_vulkan_version(raw: str) -> str:
    match = re.search(r"(?:apiVersion|vulkanVersion)\s*[:=]\s*(\d+)\.(\d+)", raw)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    match = re.search(r"1\.[0-3]", raw)
    return match.group(0) if match else "1.0"


def _extract_extensions(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    extensions: list[str] = []
    for device in data.get("devices", []) if isinstance(data, dict) else []:
        for item in device.get("device_extensions", []):
            if isinstance(item, dict):
                name = item.get("extensionName") or item.get("name")
            else:
                name = str(item)
            if name:
                extensions.append(str(name))
    return sorted(set(extensions))


def _normalize_vendor(value: str) -> str:
    normalized = value.lower()
    mapping = {
        "qualcomm": "Qualcomm",
        "adreno": "Qualcomm",
        "arm": "ARM",
        "mali": "ARM",
        "powervr": "PowerVR",
        "nvidia": "NVIDIA",
        "amd": "AMD",
        "radeon": "AMD",
        "intel": "Intel",
    }
    return mapping.get(normalized, value)


def detect_target_platform(user_declared: dict[str, Any] | None, adb_connected: bool) -> TargetContext:
    if adb_connected:
        hardware = _adb_shell("getprop ro.hardware.vulkan")
        vkjson = _adb_shell("cmd gpu vkjson")
        parsed = _parse_vkjson(vkjson)
        gpu_name = hardware or parsed.get("gpuName") or parsed.get("gpu_name") or "unknown"
        vendor = _normalize_vendor(str(parsed.get("vendorName") or parsed.get("gpuVendor") or gpu_name))
        return TargetContext(
            os="Android",
            arch=str(parsed.get("arch") or user_declared.get("arch") if user_declared else "unknown"),
            gpu_vendor=vendor,
            gpu_model=gpu_name,
            driver_version=str(parsed.get("driverVersion") or parsed.get("driver_version") or None),
            vulkan_version=_extract_vulkan_version(vkjson),
            supported_extensions=_extract_extensions(vkjson),
            unsupported_extensions=[],
            memory_heaps=list(parsed.get("memoryHeaps") or []),
            quirk_profile=get_quirk_profile(vendor, gpu_name),
            auto_detected=True,
            source="adb_query",
        )

    if user_declared:
        vendor = _normalize_vendor(str(user_declared.get("gpu_vendor") or user_declared.get("vendor") or "unknown"))
        vulkan_version = str(user_declared.get("vulkan_version") or "1.0")
        if vulkan_version not in {"1.0", "1.1", "1.2", "1.3"}:
            raise PlatformContextError("vulkan_version must be one of 1.0, 1.1, 1.2, 1.3")
        return TargetContext(
            os=str(user_declared.get("os") or "unknown"),
            arch=str(user_declared.get("arch") or "unknown"),
            gpu_vendor=vendor,
            gpu_model=str(user_declared.get("gpu_model") or "unknown"),
            driver_version=user_declared.get("driver_version"),
            vulkan_version=vulkan_version,
            supported_extensions=[str(item) for item in user_declared.get("supported_extensions", [])],
            unsupported_extensions=[str(item) for item in user_declared.get("unsupported_extensions", [])],
            memory_heaps=list(user_declared.get("memory_heaps", [])),
            quirk_profile=get_quirk_profile(vendor, str(user_declared.get("gpu_model") or "unknown")),
            auto_detected=False,
            source="user_declared",
        )

    # Fall back to host-as-target (same-platform development)
    host = detect_host_platform()
    vendor = _normalize_vendor(host.os)
    return TargetContext(
        os=host.os,
        arch=host.arch,
        gpu_vendor=vendor,
        gpu_model=host.os,
        driver_version=None,
        vulkan_version="1.0",
        supported_extensions=[],
        unsupported_extensions=[],
        memory_heaps=[],
        quirk_profile=get_quirk_profile(vendor, host.os),
        auto_detected=True,
        source="host_as_target",
    )


def determine_cross_compile(host: HostContext, target: TargetContext) -> ToolchainContext:
    is_cross = host.os != target.os or host.arch != target.arch
    if not is_cross:
        return ToolchainContext(is_cross_compile=False)

    if host.os == "Linux" and target.os == "Android":
        ndk_path = os.environ.get("NDK_HOME") or os.environ.get("ANDROID_NDK_HOME")
        compiler = f"{ndk_path}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" if ndk_path else None
        return ToolchainContext(
            is_cross_compile=True,
            cross_compiler=compiler,
            ndk_path=ndk_path,
            cmake_toolchain_file=f"{ndk_path}/build/cmake/android.toolchain.cmake" if ndk_path else None,
        )

    if host.os == "Linux" and target.os == "Windows":
        compiler = shutil.which("x86_64-w64-mingw32-clang++") or shutil.which("x86_64-w64-mingw32-g++")
        return ToolchainContext(is_cross_compile=True, cross_compiler=compiler)

    if host.os == "Windows" and target.os == "Linux":
        return ToolchainContext(
            is_cross_compile=True,
            cross_compiler=None,
            sysroot="WSL or Docker toolchain recommended",
        )

    if host.os == "Darwin" and target.os == "Android":
        ndk_path = os.environ.get("NDK_HOME") or os.environ.get("ANDROID_NDK_HOME")
        compiler = f"{ndk_path}/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++" if ndk_path else None
        return ToolchainContext(
            is_cross_compile=True,
            cross_compiler=compiler,
            ndk_path=ndk_path,
            cmake_toolchain_file=f"{ndk_path}/build/cmake/android.toolchain.cmake" if ndk_path else None,
        )

    return ToolchainContext(is_cross_compile=True, cross_compiler=None)
