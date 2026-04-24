"""Prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import PredictionRequest, PredictionResponse
from services import inference_service
from services.model_registry import KNOWN_MODELS

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    if req.model not in KNOWN_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model '{req.model}'; must be one of {list(KNOWN_MODELS)}",
        )

    try:
        result = await run_in_threadpool(inference_service.predict, req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    return PredictionResponse(
        model_name=result.model_name,
        horizon=result.horizon,
        n_sequences=result.n_sequences,
        dates=result.dates,
        actual=result.actual,
        predicted=result.predicted,
        metrics=result.metrics,
    )
