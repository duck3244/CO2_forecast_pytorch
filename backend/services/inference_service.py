"""Inference service: load checkpoints, run test-set predictions, produce API payload."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from api.config import load_config
from services.model_registry import KNOWN_MODELS, checkpoint_path
from src.data.data_loader import CO2DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.models.models import create_ensemble, create_model


_device: Optional[torch.device] = None
_data_loader: Optional[CO2DataLoader] = None
_test_loader = None
_model_cache: Dict[str, torch.nn.Module] = {}  # size ≤ 1 (evict previous)
_cache_lock = threading.Lock()


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_data_loader() -> CO2DataLoader:
    """Lazy-load the shared CO2DataLoader with current config."""
    global _data_loader, _test_loader
    if _data_loader is None:
        config = load_config()
        dl = CO2DataLoader(config)
        _, _, _test_loader, _ = dl.prepare_data()
        _data_loader = dl
    return _data_loader


def _get_test_loader():
    _get_data_loader()
    return _test_loader


def _load_model(name: str) -> torch.nn.Module:
    if name not in KNOWN_MODELS:
        raise ValueError(f"unknown model '{name}'")

    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    config = load_config()
    device = _get_device()

    if name == "ensemble":
        model = create_ensemble(config)
    else:
        model = create_model(name, config)

    # weights_only=False required for existing checkpoints (they embed config dict)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def _get_model_cached(name: str) -> torch.nn.Module:
    """LRU(size=1) model cache — evicts previous model to free GPU memory."""
    with _cache_lock:
        if name in _model_cache:
            return _model_cache[name]
        # Evict previous
        for prev_name, prev_model in list(_model_cache.items()):
            del _model_cache[prev_name]
            del prev_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = _load_model(name)
        _model_cache[name] = model
        return model


@dataclass
class PredictionResult:
    model_name: str
    dates: list[str]                 # first-step target date per sequence
    actual: list[float]              # first-step actual value per sequence
    predicted: list[float]           # first-step predicted value per sequence
    metrics: dict[str, float]
    horizon: int
    n_sequences: int


def predict(model_name: str) -> PredictionResult:
    model = _get_model_cached(model_name)
    data_loader = _get_data_loader()
    test_loader = _get_test_loader()
    device = _get_device()

    evaluator = ModelEvaluator(
        model=model,
        data_loader=test_loader,
        device=device,
        scaler=data_loader,
        split="test",
    )
    results = evaluator.evaluate()

    predictions = np.asarray(results["predictions"])  # (N, H) or (N,)
    targets = np.asarray(results["targets"])

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    first_pred = predictions[:, 0].astype(float).tolist()
    first_actual = targets[:, 0].astype(float).tolist()

    test_dates = data_loader._target_dates.get("test", [])
    first_dates = [d[0].strftime("%Y-%m-%d") for d in test_dates[: len(first_pred)]]

    metrics = {k: float(v) for k, v in results["metrics"].items()}

    return PredictionResult(
        model_name=model_name,
        dates=first_dates,
        actual=first_actual,
        predicted=first_pred,
        metrics=metrics,
        horizon=predictions.shape[1],
        n_sequences=predictions.shape[0],
    )


def unload_all() -> None:
    """Free cached model (called by training service before starting a new job)."""
    with _cache_lock:
        _model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
