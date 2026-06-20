from __future__ import annotations

from pathlib import Path

from . import SubprocessResult, _run


def run_glslang_validator(shader_file: str | Path) -> SubprocessResult:
    return _run(["glslangValidator", "-V", str(shader_file)])
