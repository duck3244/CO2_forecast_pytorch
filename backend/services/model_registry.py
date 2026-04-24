"""Discovery and metadata for trained model checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

from api.config import load_config, resolve_path
from api.schemas import ModelInfo

KNOWN_MODELS = ("lstm", "transformer", "hybrid", "ensemble")


def checkpoint_path(model_name: str) -> Path:
    config = load_config()
    model_dir = resolve_path(config["paths"]["model_dir"])
    return model_dir / f"{model_name}_model.pth"


def list_models() -> List[ModelInfo]:
    results: List[ModelInfo] = []
    for name in KNOWN_MODELS:
        path = checkpoint_path(name)
        if path.exists():
            stat = path.stat()
            results.append(
                ModelInfo(
                    name=name,
                    trained=True,
                    file_size_bytes=stat.st_size,
                    saved_at=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    checkpoint_path=str(path),
                )
            )
        else:
            results.append(ModelInfo(name=name, trained=False))
    return results


def device_info() -> tuple[str, Optional[str]]:
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", None
