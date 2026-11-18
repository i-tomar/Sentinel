"""In-process run store (memory). Lost on process exit. Cap with oldest-first eviction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

from sentinel.api.models import RunStatus, StageEvent
from sentinel.graph.build import AskResult

DEFAULT_MAX_RUNS = 200


@dataclass
class StoredRun:
    run_id: str
    status: RunStatus
    result: AskResult
    events: list[StageEvent] = field(default_factory=list)


class RunStore:
    def __init__(self, *, max_runs: int = DEFAULT_MAX_RUNS) -> None:
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        self.max_runs = max_runs
        self._runs: OrderedDict[str, StoredRun] = OrderedDict()
        self._lock = Lock()

    def put(self, run: StoredRun) -> None:
        with self._lock:
            if run.run_id in self._runs:
                del self._runs[run.run_id]
            self._runs[run.run_id] = run
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> StoredRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)
