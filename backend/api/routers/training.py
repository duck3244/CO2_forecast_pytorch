"""Training job endpoints with SSE event streaming."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    TrainingJobRequest,
    TrainingJobSnapshot,
    TrainingJobsResponse,
)
from api.state import registry
from services import training_service
from services.model_registry import KNOWN_MODELS

router = APIRouter(prefix="/api/training", tags=["training"])

# Poll interval for SSE event buffer (events are produced by another thread).
_POLL_INTERVAL = 0.3
# Heartbeat so proxies / browsers keep the connection alive while training.
_HEARTBEAT_INTERVAL = 15.0


@router.post("/jobs", status_code=202)
def create_job(req: TrainingJobRequest) -> dict:
    if req.model not in KNOWN_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model '{req.model}'; must be among {list(KNOWN_MODELS)}",
        )
    overrides = req.overrides.model_dump(exclude_none=True) if req.overrides else {}
    job = registry.create(model=req.model, overrides=overrides)
    training_service.submit(job)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs", response_model=TrainingJobsResponse)
def list_jobs() -> TrainingJobsResponse:
    snaps = [TrainingJobSnapshot(**j.to_snapshot()) for j in registry.list()]
    snaps.sort(key=lambda s: s.created_at, reverse=True)
    return TrainingJobsResponse(jobs=snaps)


@router.get("/jobs/{job_id}", response_model=TrainingJobSnapshot)
def get_job(job_id: str) -> TrainingJobSnapshot:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return TrainingJobSnapshot(**job.to_snapshot())


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.is_terminal():
        return {"id": job.id, "status": job.status, "cancelled": False}
    job.cancel_event.set()
    return {"id": job.id, "status": job.status, "cancelled": True}


async def _event_stream(
    job_id: str, last_event_id: Optional[int], request: Request
) -> AsyncIterator[dict]:
    job = registry.get(job_id)
    if job is None:
        yield {"event": "error", "data": json.dumps({"message": "job not found"})}
        return

    cursor = last_event_id if last_event_id is not None else -1
    last_heartbeat = asyncio.get_event_loop().time()

    while True:
        if await request.is_disconnected():
            return

        new_events = await asyncio.to_thread(job.events_since, cursor)
        for e in new_events:
            yield {
                "id": str(e.id),
                "event": e.type,
                "data": json.dumps(e.data),
            }
            cursor = e.id
            last_heartbeat = asyncio.get_event_loop().time()

        if job.is_terminal() and not job.events_since(cursor):
            # Terminal summary — emit once so clients can clean up
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "status": job.status,
                        "stopped_reason": job.stopped_reason,
                        "error": job.error,
                    }
                ),
            }
            return

        await asyncio.to_thread(job.wait_for_change, cursor, _POLL_INTERVAL)

        now = asyncio.get_event_loop().time()
        if now - last_heartbeat > _HEARTBEAT_INTERVAL:
            yield {"event": "heartbeat", "data": "{}"}
            last_heartbeat = now


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> EventSourceResponse:
    last_id_header = request.headers.get("last-event-id")
    last_id = None
    if last_id_header is not None:
        try:
            last_id = int(last_id_header)
        except ValueError:
            pass
    return EventSourceResponse(_event_stream(job_id, last_id, request))
