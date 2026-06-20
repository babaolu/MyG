from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

COMMAND_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class SubprocessResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    fault: str | None = None


def _run(command: Sequence[str], cwd: Path | None = None) -> SubprocessResult:
    executable = shutil.which(command[0])
    if executable is None:
        return SubprocessResult(
            command=list(command),
            returncode=127,
            stdout="",
            stderr=f"command not found: {command[0]}",
            fault=f"command not found: {command[0]}",
        )
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired as exc:
        return SubprocessResult(
            command=list(command),
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            fault=f"command timed out after {COMMAND_TIMEOUT_SECONDS} seconds",
        )
    fault = None
    if completed.returncode != 0:
        fault = completed.stderr or completed.stdout or f"exit code {completed.returncode}"
    return SubprocessResult(
        command=list(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        fault=fault,
    )
