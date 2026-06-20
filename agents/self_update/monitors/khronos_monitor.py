from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field


class UpdateDiff(BaseModel):
    update_id: str
    source: str
    current_version: str
    latest_version: str
    changelog: str
    new_extensions: list[str] = Field(default_factory=list)
    deprecated_patterns: list[str] = Field(default_factory=list)
    require_human_confirmation: bool = True
    confirmed: bool | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class KhronosMonitor:
    def __init__(self, current_spec_version: str) -> None:
        self.current_spec_version = current_spec_version

    def poll(self) -> UpdateDiff | None:
        response = httpx.get("https://api.github.com/repos/KhronosGroup/Vulkan-Docs/releases/latest", timeout=30)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        latest = str(data.get("tag_name") or "").lstrip("v")
        if not latest or latest == self.current_spec_version:
            return None
        changelog = str(data.get("body") or "")
        return UpdateDiff(
            update_id=f"khronos-{latest}",
            source="khronos_vulkan_docs",
            current_version=self.current_spec_version,
            latest_version=latest,
            changelog=changelog,
            new_extensions=_extract_extensions(changelog),
            deprecated_patterns=_extract_deprecated(changelog),
        )


def _extract_extensions(changelog: str) -> list[str]:
    return sorted({part.strip(" `.,;") for part in changelog.split() if part.startswith("VK_")})


def _extract_deprecated(changelog: str) -> list[str]:
    lowered = changelog.lower()
    return ["deprecated Vulkan extension" if "deprecat" in lowered else ""] if "deprecat" in lowered else []
