"""Who may write long-term memory, and how a backend is chosen.

Agent writes (saving a fact, marking one stale) are write operations:
- live data: always pending human approval (the approval flow lands with the permission gate);
- fixture/eval data: pending too, unless auto-approve was explicitly enabled.
`Settings` refuses MEMORY_AUTO_APPROVE together with DATA_MODE=live, and so does `decide`.
"""

from __future__ import annotations

from typing import Literal

from sentinel.config import Settings
from sentinel.memory.backend import MemoryBackend

WriteDecision = Literal["allow", "require_approval"]


def decide(data_mode: Literal["fixture", "live"], auto_approve: bool) -> WriteDecision:
    if data_mode == "live":
        if auto_approve:
            raise ValueError("memory auto-approve is only allowed with DATA_MODE=fixture")
        return "require_approval"
    return "allow" if auto_approve else "require_approval"


def make_memory_backend(settings: Settings) -> MemoryBackend:
    if settings.memory_backend == "okf":
        raise NotImplementedError(
            "MEMORY_BACKEND=okf is a stretch goal and not implemented yet; use sqlite"
        )
    from sentinel.memory.sqlite_backend import SQLiteMemoryBackend

    return SQLiteMemoryBackend(settings.memory_db)
