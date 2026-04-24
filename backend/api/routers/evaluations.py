"""Multi-model evaluation/comparison endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    PerModelEvaluation,
)
from services import inference_service
from services.model_registry import KNOWN_MODELS

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


def _evaluate_sync(models: list[str]) -> EvaluationResponse:
    results: list[PerModelEvaluation] = []
    shared_dates: list[str] = []
    shared_actual: list[float] = []
    horizon = 0
    n_sequences = 0

    for i, name in enumerate(models):
        pred = inference_service.predict(name)
        if i == 0:
            shared_dates = pred.dates
            shared_actual = pred.actual
            horizon = pred.horizon
            n_sequences = pred.n_sequences
        results.append(
            PerModelEvaluation(
                model_name=pred.model_name,
                predicted=pred.predicted,
                metrics=pred.metrics,
            )
        )

    best_by_r2 = None
    best_score = float("-inf")
    for r in results:
        score = r.metrics.get("R2")
        if score is not None and score > best_score:
            best_score = score
            best_by_r2 = r.model_name

    return EvaluationResponse(
        horizon=horizon,
        n_sequences=n_sequences,
        dates=shared_dates,
        actual=shared_actual,
        results=results,
        best_by_r2=best_by_r2,
    )


@router.post("", response_model=EvaluationResponse)
async def evaluate(req: EvaluationRequest) -> EvaluationResponse:
    unknown = [m for m in req.models if m not in KNOWN_MODELS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown models: {unknown}; must be among {list(KNOWN_MODELS)}",
        )

    try:
        return await run_in_threadpool(_evaluate_sync, req.models)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation failed: {e}")
