"""Per-run log files and a lock on the output directory.

Every `python -m evals.run` gets a run id (UTC start time, model, subset, process id) and its
own log, `<out>/logs/eval-<run id>.log`. Progress lines, errors and the outcome go to the
console and the log, each flushed as it is written. A later run never overwrites an earlier
run's log, so a repeated or interrupted run stays visible.

`exclusive_run` holds an OS lock on `<out>/.eval.lock` for the whole run, so two runs can't
write the same report directory at once. The OS releases it when the process exits, even if
the process is killed, so a stale lock can't block the next run.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO

LOCK_NAME = ".eval.lock"


class RunInProgressError(RuntimeError):
    pass


def run_id(started: datetime, model: str, subset: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in model.lower())
    return f"{started:%Y%m%dT%H%M%SZ}-{slug}-{subset}-p{os.getpid()}"


def log_path(out: Path, rid: str) -> Path:
    return out / "logs" / f"eval-{rid}.log"


class RunLog:
    """Writes each line to the console and the run's log file, flushing both."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: IO[str] = path.open("a", encoding="utf-8")

    def line(self, message: str, *, err: bool = False) -> None:
        print(message, file=sys.stderr if err else sys.stdout, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _try_lock(handle: IO[str]) -> bool:
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[str]) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_run(out: Path) -> Iterator[None]:
    """Hold the output directory for one run; raise `RunInProgressError` if another run has it."""
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / LOCK_NAME).open("a+", encoding="utf-8")
    if not _try_lock(handle):
        handle.close()
        raise RunInProgressError(
            f"another eval run is writing to {out} (lock {out / LOCK_NAME}); wait for it to "
            "finish or use a different --out"
        )
    try:
        yield
    finally:
        _unlock(handle)
        handle.close()
