from __future__ import annotations

from pathlib import Path

from . import SubprocessResult, _run


def run_spirv_val(spirv_file: str | Path) -> SubprocessResult:
    return _run(["spirv-val", "--target-env", "vulkan1.3", str(spirv_file)])
