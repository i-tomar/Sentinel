"""Local, per-source storage for fixture mutations (approved writes).

The generated fixture data under `servers/fixtures/data/` is never written. A server started
with a state directory loads `<source>.json` from it on startup, applies it on top of the
generated records, and rewrites it after every successful write, so the change survives into
the next process (the next CLI command). Without a directory, mutations live in memory only.

The file is rewritten whole (temp file + atomic rename). Two server processes writing the same
source at the same time are not coordinated: the last write wins.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

STATE_DIR_ENV = "SENTINEL_FIXTURE_STATE_DIR"


def state_dir_from_env() -> Path | None:
    value = os.environ.get(STATE_DIR_ENV, "").strip()
    return Path(value) if value else None


class FixtureState:
    def __init__(self, source: str, directory: Path | None) -> None:
        self.path = directory / f"{source}.json" if directory is not None else None

    def load(self) -> dict[str, Any]:
        if self.path is None or not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def save(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.stem}-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
