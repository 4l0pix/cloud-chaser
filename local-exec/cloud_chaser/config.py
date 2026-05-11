from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and optionally apply nested dictionary overrides."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(deepcopy(cfg), overrides)
    return cfg


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_device(cfg: dict[str, Any]) -> str:
    import torch

    requested = cfg.get("project", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested
