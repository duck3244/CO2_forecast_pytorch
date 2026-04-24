"""Training orchestration: wraps `Trainer` and emits SSE events via JobRegistry.

Single worker (max_workers=1) so only one model trains at a time — keeps
GPU memory bounded and serializes logs on the SSE stream.
"""
from __future__ import annotations

import copy
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import torch

from api.config import load_config, resolve_path
from api.state import TrainingJob
from services import inference_service
from services.model_registry import KNOWN_MODELS, checkpoint_path
from src.data.data_loader import CO2DataLoader
from src.models.models import create_ensemble, create_model
from src.training.trainer import EnsembleTrainer, Trainer

# Single-worker pool: MVP requirement (no concurrent training).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")
_run_lock = threading.Lock()


def submit(job: TrainingJob) -> Future:
    """Submit a job to the single training worker. Returns the Future."""
    return _executor.submit(_run_job, job)


def _apply_overrides(base_config: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    for k in ("epochs", "learning_rate", "batch_size", "patience", "weight_decay"):
        if k in overrides and overrides[k] is not None:
            cfg["training"][k] = overrides[k]
    return cfg


def _make_progress_callback(job: TrainingJob):
    def _cb(epoch, total_epochs, train_loss, val_loss, val_metrics, best_val_loss):
        job.push(
            "progress",
            {
                "epoch": int(epoch),
                "total_epochs": int(total_epochs),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val_loss": float(best_val_loss),
                "val_metrics": {k: float(v) for k, v in (val_metrics or {}).items()},
            },
        )
    return _cb


def _run_job(job: TrainingJob) -> None:
    """Runs inside the training thread."""
    with _run_lock:
        try:
            job.mark_running()
            job.push("log", {"message": f"starting training for '{job.model}'"})

            # Evict inference model cache to avoid GPU OOM
            inference_service.unload_all()

            base_config = load_config()
            config = _apply_overrides(base_config, job.overrides or {})
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            job.push("log", {"message": f"device: {device}"})

            data_loader = CO2DataLoader(config)
            train_loader, val_loader, _, _ = data_loader.prepare_data()
            job.push("log", {"message": "data prepared"})

            if job.model in ("lstm", "transformer", "hybrid"):
                model = create_model(job.model, config)
                trainer = Trainer(model, train_loader, val_loader, config, device)
                result = trainer.train(
                    progress_callback=_make_progress_callback(job),
                    cancel_flag=job.cancel_event,
                )
                save_path = checkpoint_path(job.model)
                trainer.save_model(str(save_path))

            elif job.model == "ensemble":
                # Load pretrained individual models if present; otherwise train each.
                individual_models = []
                ensemble_cfg = config["ensemble"]["models"]
                all_pretrained = True
                for name in ensemble_cfg:
                    path = checkpoint_path(name)
                    m = create_model(name, config)
                    if path.exists():
                        ckpt = torch.load(path, map_location=device, weights_only=False)
                        m.load_state_dict(ckpt["model_state_dict"])
                        job.push("log", {"message": f"loaded pretrained {name}"})
                    else:
                        all_pretrained = False
                        job.push("log", {"message": f"{name} not pretrained — will train"})
                    individual_models.append(m)

                ens_trainer = EnsembleTrainer(
                    individual_models,
                    train_loader,
                    val_loader,
                    config,
                    device,
                    already_trained=all_pretrained,
                )
                ens_trainer.fine_tune_ensemble()
                save_path = checkpoint_path("ensemble")
                torch.save(
                    {
                        "model_state_dict": ens_trainer.ensemble_model.state_dict(),
                        "config": config,
                    },
                    save_path,
                )
                result = {
                    "train_losses": [],
                    "val_losses": [],
                    "best_val_loss": float("nan"),
                    "training_time": 0.0,
                    "stopped_reason": "completed",
                }
            else:
                raise ValueError(f"unknown model '{job.model}'")

            # Invalidate inference cache so next /api/predictions picks up the new weights
            inference_service.unload_all()

            reason = result.get("stopped_reason", "completed")
            status = "cancelled" if reason == "cancelled" else "completed"
            job.push(
                "completed",
                {
                    "reason": reason,
                    "best_val_loss": float(result.get("best_val_loss", float("nan"))),
                    "training_time_s": float(result.get("training_time", 0.0)),
                    "saved_to": str(save_path),
                },
            )
            job.mark_done(status=status, reason=reason)

        except Exception as e:
            tb = traceback.format_exc()
            job.push("error", {"message": str(e), "traceback": tb})
            job.mark_done(status="failed", error=str(e))
