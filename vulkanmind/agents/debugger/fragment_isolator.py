from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FragmentSpec:
    source_file: str
    subsystem: str
    start_line: int
    end_line: int
    isolated_text: str


def extract_fragment(source_file: str | Path, suspected_subsystem: str) -> FragmentSpec:
    path = Path(source_file)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lowered = suspected_subsystem.lower()
    start = 0
    end = len(lines)
    for index, line in enumerate(lines):
        if lowered in line.lower():
            start = index
            break
    brace_depth = 0
    saw_brace = False
    for index in range(start, len(lines)):
        line = lines[index]
        brace_depth += line.count("{") - line.count("}")
        if "{" in line:
            saw_brace = True
        if saw_brace and brace_depth <= 0 and index > start:
            end = index + 1
            break
    return FragmentSpec(
        source_file=str(path),
        subsystem=suspected_subsystem,
        start_line=start + 1,
        end_line=end,
        isolated_text="\n".join(lines[start:end]),
    )
