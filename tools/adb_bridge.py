from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any


class ADBUnavailable(RuntimeError):
    pass


def adb_shell(command: str, timeout: int = 15) -> str:
    executable = shutil.which("adb")
    if executable is None:
        raise ADBUnavailable("adb is not available")
    completed = subprocess.run(
        [executable, "shell", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout


def query_gpu_vkjson() -> dict[str, Any]:
    raw = adb_shell("cmd gpu vkjson")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("adb cmd gpu vkjson did not return JSON") from exc


def query_hardware_vulkan() -> str:
    return adb_shell("getprop ro.hardware.vulkan").strip()
