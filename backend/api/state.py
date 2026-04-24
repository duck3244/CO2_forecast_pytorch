"""In-memory job registry for training jobs. Lost on server restart (MVP)."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class JobEvent:
    id: int
    ts: float
    type: str  # 'progress' | 'log' | 'completed' | 'error'
    data: dict


@dataclass
class TrainingJob:
    id: str
    model: str
    overrides: dict
    status: str = "queued"  # queued | running | completed | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    stopped_reason: Optional[str] = None
    error: Optional[str] = None
    events: List[JobEvent] = field(default_factory=list)
    _next_event_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _condition: threading.Condition = field(init=False)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)

    # -- producer side (training thread) --
    def push(self, type_: str, data: dict) -> None:
        with self._condition:
            event = JobEvent(
                id=self._next_event_id,
                ts=time.time(),
                type=type_,
                data=data,
            )
            self._next_event_id += 1
            self.events.append(event)
            self._condition.notify_all()

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = time.time()

    def mark_done(self, status: str, error: Optional[str] = None, reason: Optional[str] = None) -> None:
        with self._condition:
            self.status = status
            self.error = error
            self.stopped_reason = reason
            self.finished_at = time.time()
            self._condition.notify_all()

    # -- consumer side (SSE generator) --
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def events_since(self, last_id: int) -> List[JobEvent]:
        """Return events with id > last_id (thread-safe snapshot)."""
        with self._lock:
            return [e for e in self.events if e.id > last_id]

    def wait_for_change(self, last_id: int, timeout: float) -> None:
        """Block until new event available (id > last_id) OR job terminal OR timeout."""
        with self._condition:
            def ready() -> bool:
                return (self.events and self.events[-1].id > last_id) or self.is_terminal()
            if ready():
                return
            self._condition.wait(timeout=timeout)

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "overrides": self.overrides,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stopped_reason": self.stopped_reason,
            "error": self.error,
            "n_events": len(self.events),
        }


class JobRegistry:
    """Process-wide singleton. Jobs are kept for the lifetime of the process."""

    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create(self, model: str, overrides: dict) -> TrainingJob:
        job = TrainingJob(id=uuid.uuid4().hex, model=model, overrides=overrides)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[TrainingJob]:
        with self._lock:
            return list(self._jobs.values())


# Module-level singleton
registry = JobRegistry()
