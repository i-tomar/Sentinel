"""Per-run log files and a lock on the output directory.

Every `python -m evals.run` gets a run id (UTC start time to the microsecond, model, subset,
process id, random suffix) and its own log, `<out>/logs/eval-<run id>.log`. The log is created
in exclusive mode, so a run never appends to or overwrites an earlier run's log, even two runs
in the same process within the same second; a repeated or interrupted run stays visible.
Progress lines, errors and the outcome go to the console and the log, each flushed as written.

`exclusive_run` holds an OS lock on `<out>/.eval.lock` for the whole run, so two runs can't
write the same report directory at once. The OS releases it when the process exits, even if
the process is killed, so a stale lock can't block the next run.
"""

from __future__ import annotations

import os
import secrets
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
    stamp = f"{started:%Y%m%dT%H%M%S}{started.microsecond:06d}Z"
    return f"{stamp}-{slug}-{subset}-p{os.getpid()}-{secrets.token_hex(3)}"


def log_path(out: Path, rid: str) -> Path:
    return out / "logs" / f"eval-{rid}.log"


class RunLog:
    """Writes each line to the console and the run's log file, flushing both. The file is
    created in exclusive mode: `FileExistsError` if it already exists, never an overwrite."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: IO[str] = path.open("x", encoding="utf-8")

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


def new_run_log(
    out: Path, started: datetime, model: str, subset: str, attempts: int = 5
) -> tuple[str, RunLog]:
    """A fresh run id and its newly created log. Picks a new id in the (practically
    impossible) case that the file already exists, instead of touching it."""
    for _ in range(attempts):
        rid = run_id(started, model, subset)
        try:
            return rid, RunLog(log_path(out, rid))
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a new run log in {out / 'logs'}")


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
