"""Optional Open Knowledge Format (OKF) file backend. Stretch goal; not implemented yet."""

from __future__ import annotations

from pathlib import Path


class OKFMemoryBackend:
    def __init__(self, directory: Path) -> None:
        raise NotImplementedError(
            "MEMORY_BACKEND=okf is a stretch goal and not implemented yet; use sqlite"
        )
