from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH: Path = Path(os.environ.get("VULKANMIND_CONFIG", "config.yaml"))


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the VulkanMind YAML configuration.

    A pure-defaults return value (`{}`) when the file is absent allows tests and
    CLI tooling to operate without a config present.
    """
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
