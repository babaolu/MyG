from __future__ import annotations

from agents.code_generation.android_cmake import generate_cmake_snippet as android_snippet
from agents.code_generation.embedded_cmake import generate_cmake_snippet as embedded_snippet
from agents.code_generation.linux_cmake import generate_cmake_snippet as linux_snippet
from agents.code_generation.windows_cmake import generate_cmake_snippet as windows_snippet
from orchestrator.state import PlatformContext


def cmake_for_platform(context: PlatformContext) -> str:
    target_os = context.target.os.lower()
    if target_os == "android":
        return android_snippet(context)
    if target_os == "windows":
        return windows_snippet(context)
    if target_os in {"linux", "freebsd", "openbsd", "netbsd"}:
        return linux_snippet(context)
    return embedded_snippet(context)
