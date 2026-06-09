"""Configuration helpers for scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/default.json") -> dict[str, Any]:
    """Load a JSON config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_dirs(*paths: str | Path) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(data: Any, path: str | Path) -> None:
    """Write indented JSON with UTF-8 encoding."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
