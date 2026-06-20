from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def parse_renderdoc_capture(path: str | Path) -> dict[str, Any]:
    capture_path = Path(path)
    text = capture_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "capture_path": str(capture_path),
        "api_calls": len(re.findall(r"vk[A-Z][A-Za-z0-9_]+", text)),
        "validation_markers": sorted(set(re.findall(r"VALIDATION|VUID-[A-Za-z0-9_]+", text))),
        "raw_excerpt": text[:4000],
    }


def parse_renderdoc_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "capture_path": str(path),
        "api_calls": len(data.get("draws", [])),
        "validation_markers": data.get("validation", []),
        "raw_excerpt": json.dumps(data)[:4000],
    }
