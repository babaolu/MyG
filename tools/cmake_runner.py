from __future__ import annotations

from pathlib import Path

from . import SubprocessResult, _run


def configure_cmake(source_dir: str | Path, build_dir: str | Path, toolchain_file: str | Path | None = None) -> SubprocessResult:
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if toolchain_file:
        command.extend(["-DCMAKE_TOOLCHAIN_FILE", str(toolchain_file)])
    return _run(command)


def build_cmake(build_dir: str | Path, parallel_workers: int | None = None) -> SubprocessResult:
    command = ["cmake", "--build", str(build_dir)]
    if parallel_workers is not None and parallel_workers > 0:
        command.extend(["--parallel", str(parallel_workers)])
    return _run(command)
