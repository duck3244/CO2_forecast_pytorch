"""Smoke tests for the FastAPI surface.

Covers the happy path for read-only endpoints and the training job lifecycle
(create → SSE stream → completion) with minimal epochs.
"""
from __future__ import annotations

import time

import pytest


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert set(m["name"] for m in body["models"]) == {
        "lstm",
        "transformer",
        "hybrid",
        "ensemble",
    }
    assert body["device"] in ("cpu", "cuda")


def test_dataset(client):
    r = client.get("/api/datasets/co2")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in ("cache", "noaa")
    assert body["n_records"] > 500
    assert len(body["dates"]) == body["n_records"]
    assert len(body["values"]) == body["n_records"]


@pytest.mark.parametrize("model", ["lstm", "transformer", "hybrid", "ensemble"])
def test_predict_each_model(client, model):
    r = client.post("/api/predictions", json={"model": model})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_name"] == model
    assert body["horizon"] >= 1
    assert body["n_sequences"] == len(body["dates"]) == len(body["actual"]) == len(body["predicted"])
    assert "R2" in body["metrics"]


def test_evaluate_multi_model(client):
    r = client.post("/api/evaluations", json={"models": ["lstm", "transformer"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 2
    assert {x["model_name"] for x in body["results"]} == {"lstm", "transformer"}
    assert body["best_by_r2"] in {"lstm", "transformer"}


def test_predict_unknown_model_returns_400(client):
    r = client.post("/api/predictions", json={"model": "does-not-exist"})
    assert r.status_code == 400


def test_training_job_lifecycle(client):
    # Kick off a 2-epoch LSTM training job
    r = client.post(
        "/api/training/jobs",
        json={"model": "lstm", "overrides": {"epochs": 2, "patience": 5}},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    # Poll snapshot until terminal or timeout
    deadline = time.time() + 30
    while time.time() < deadline:
        snap = client.get(f"/api/training/jobs/{job_id}").json()
        if snap["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)

    assert snap["status"] == "completed", snap
    assert snap["stopped_reason"] in ("completed", "early_stopped")
    assert snap["n_events"] >= 2  # at least one progress + completed


def test_sse_stream_emits_progress_and_completion(client):
    r = client.post(
        "/api/training/jobs",
        json={"model": "lstm", "overrides": {"epochs": 2, "patience": 5}},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    events_seen: list[str] = []
    with client.stream(
        "GET",
        f"/api/training/jobs/{job_id}/events",
        headers={"Accept": "text/event-stream"},
    ) as stream:
        for line in stream.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                events_seen.append(line.split(":", 1)[1].strip())
            # SSE stream ends with 'done' event — break once observed
            if events_seen and events_seen[-1] == "done":
                break

    assert "progress" in events_seen
    assert "completed" in events_seen
    assert events_seen[-1] == "done"


def test_training_cancel(client):
    r = client.post(
        "/api/training/jobs",
        json={"model": "lstm", "overrides": {"epochs": 10, "patience": 20}},
    )
    job_id = r.json()["job_id"]
    # Give it a moment to start
    time.sleep(1.0)
    cancel = client.delete(f"/api/training/jobs/{job_id}").json()
    assert cancel["cancelled"] is True

    deadline = time.time() + 20
    while time.time() < deadline:
        snap = client.get(f"/api/training/jobs/{job_id}").json()
        if snap["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.3)
    # Cancellation is cooperative at epoch boundary — may show completed if it finished first
    assert snap["status"] in ("cancelled", "completed")
