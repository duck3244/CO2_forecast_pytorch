"""Pydantic schemas for API I/O."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# --- datasets ---


class CO2Dataset(BaseModel):
    source: str = Field(..., description="'noaa' | 'sample' | 'cache'")
    n_records: int
    start_date: str
    end_date: str
    dates: List[str]
    values: List[float]


# --- models ---


class ModelInfo(BaseModel):
    name: str
    trained: bool
    file_size_bytes: Optional[int] = None
    saved_at: Optional[str] = None  # ISO-8601 UTC
    checkpoint_path: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    device: str  # 'cuda' | 'cpu'
    device_name: Optional[str] = None


# --- predictions ---


class PredictionRequest(BaseModel):
    model: str = Field(..., description="lstm | transformer | hybrid | ensemble")


class PredictionResponse(BaseModel):
    model_name: str
    horizon: int
    n_sequences: int
    dates: List[str]
    actual: List[float]
    predicted: List[float]
    metrics: Dict[str, float]


# --- evaluations (multi-model comparison) ---


class EvaluationRequest(BaseModel):
    models: List[str] = Field(..., min_length=1)


class PerModelEvaluation(BaseModel):
    model_name: str
    predicted: List[float]
    metrics: Dict[str, float]


class EvaluationResponse(BaseModel):
    horizon: int
    n_sequences: int
    dates: List[str]
    actual: List[float]
    results: List[PerModelEvaluation]
    best_by_r2: Optional[str] = None


# --- training jobs ---


class TrainingOverrides(BaseModel):
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    patience: Optional[int] = None
    weight_decay: Optional[float] = None


class TrainingJobRequest(BaseModel):
    model: str = Field(..., description="lstm | transformer | hybrid | ensemble")
    overrides: Optional[TrainingOverrides] = None


class TrainingJobSnapshot(BaseModel):
    id: str
    model: str
    overrides: TrainingOverrides
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    stopped_reason: Optional[str] = None
    error: Optional[str] = None
    n_events: int


class TrainingJobsResponse(BaseModel):
    jobs: List[TrainingJobSnapshot]
