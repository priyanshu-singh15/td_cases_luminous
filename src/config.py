from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str, cfg: dict[str, Any] | None = None) -> Path:
    base = ROOT
    return (base / relative).resolve()
