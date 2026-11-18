"""In-process run store (memory). Lost on process exit."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from sentinel.api.models import RunStatus, StageEvent
from sentinel.graph.build import AskResult


@dataclass
class StoredRun:
    run_id: str
    status: RunStatus
    result: AskResult
    events: list[StageEvent] = field(default_factory=list)


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, StoredRun] = {}
        self._lock = Lock()

    def put(self, run: StoredRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get(self, run_id: str) -> StoredRun | None:
        with self._lock:
            return self._runs.get(run_id)
