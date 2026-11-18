"""Process-local sliding-window rate limit for POST /api/runs."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, *, max_calls: int = 30, window_s: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window_s:
            q.popleft()

    def _drop_idle_keys(self, now: float) -> None:
        idle: list[str] = []
        for key, q in self._hits.items():
            self._prune(q, now)
            if not q:
                idle.append(key)
        for key in idle:
            del self._hits[key]

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._drop_idle_keys(now)
            q = self._hits[key]
            self._prune(q, now)
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            return True

    def retry_after_s(self, key: str) -> int:
        """Seconds until the oldest hit in the window expires (at least 1)."""
        now = time.monotonic()
        with self._lock:
            self._drop_idle_keys(now)
            q = self._hits[key]
            self._prune(q, now)
            if len(q) < self.max_calls or not q:
                if not q and key in self._hits:
                    del self._hits[key]
                return 1
            wait = self.window_s - (now - q[0])
            return max(1, math.ceil(wait))
