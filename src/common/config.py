"""Loads configs/pipeline.yaml and resolves paths against the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"
FEATURE_REGISTRY = PROJECT_ROOT / "configs" / "feature_registry.yaml"


class Config(dict):
    """A plain dict with a helper for the folders listed under `paths`."""

    def path(self, key: str, *parts: str) -> Path:
        """Folder from `paths`, optionally a subfolder of it; created if missing."""
        p = PROJECT_ROOT / self["paths"][key]
        for part in parts:
            p = p / part
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self["project"]["seed"])


def load_config(config_path: str | Path | None = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return Config(raw)


def load_feature_registry() -> dict[str, Any]:
    with open(FEATURE_REGISTRY, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
