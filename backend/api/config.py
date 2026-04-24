"""Shared config loader. Uses backend/configs/config.yaml by default."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import yaml

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BACKEND_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def backend_root() -> Path:
    return BACKEND_ROOT


def resolve_path(relative: str) -> Path:
    """Resolve a config path relative to backend/ root."""
    p = Path(relative)
    return p if p.is_absolute() else BACKEND_ROOT / p
