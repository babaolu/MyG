from __future__ import annotations

from pathlib import Path

from . import SubprocessResult, _run


def run_clang_tidy(source_file: str | Path, build_dir: str | Path | None = None, config_file: str | Path | None = None) -> SubprocessResult:
    command = ["clang-tidy", str(source_file)]
    if build_dir:
        command.extend(["-p", str(build_dir)])
    if config_file:
        command.extend(["--config-file", str(config_file)])
    return _run(command)
