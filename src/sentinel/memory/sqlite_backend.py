"""Default long-term memory backend: one SQLite table, standard library only."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sentinel.guards.redaction import redact_memory_item
from sentinel.memory.backend import rank
from sentinel.memory.models import MemoryItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    verified INTEGER NOT NULL,
    stale INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    confidence REAL NOT NULL,
    record_keys TEXT NOT NULL,
    stale_reason TEXT,
    superseded_by TEXT,
    written_at TEXT NOT NULL
)
"""
_COLUMNS = (
    "key, value, source, verified, stale, updated_at, confidence, record_keys, "
    "stale_reason, superseded_by"
)


class SQLiteMemoryBackend:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            with self._memory_conn:
                yield self._memory_conn
            return
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(row: tuple[object, ...]) -> MemoryItem:
        (key, value, source, verified, stale, updated_at, confidence, record_keys,
         stale_reason, superseded_by) = row  # fmt: skip
        return MemoryItem(
            key=str(key),
            value=str(value),
            source=str(source),
            verified=bool(verified),
            stale=bool(stale),
            updated_at=datetime.fromisoformat(str(updated_at)),
            confidence=float(confidence),  # type: ignore[arg-type]
            record_keys=json.loads(str(record_keys)),
            stale_reason=None if stale_reason is None else str(stale_reason),
            superseded_by=None if superseded_by is None else str(superseded_by),
        )

    def get(self, key: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM memory_items WHERE key = ?",  # noqa: S608 - constant
                (key,),
            ).fetchone()
        return None if row is None else self._row(row)

    def put(self, item: MemoryItem) -> MemoryItem:
        """Insert or replace. The value and stale reason are PII-redacted first, whoever the
        writer is (agent, CLI, memory MCP server): long-term memory never holds raw PII."""
        item = redact_memory_item(item)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_items "  # noqa: S608 - constant column list
                f"({_COLUMNS}, written_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.key,
                    item.value,
                    item.source,
                    int(item.verified),
                    int(item.stale),
                    item.updated_at.isoformat(),
                    item.confidence,
                    json.dumps(item.record_keys),
                    item.stale_reason,
                    item.superseded_by,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return item

    def mark_stale(
        self, key: str, reason: str, superseded_by: str | None = None
    ) -> MemoryItem | None:
        item = self.get(key)
        if item is None:
            return None
        updated = item.model_copy(
            update={"stale": True, "stale_reason": reason[:300], "superseded_by": superseded_by}
        )
        return self.put(updated)

    def all(self, *, include_stale: bool = True) -> list[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM memory_items ORDER BY key"  # noqa: S608 - constant
            ).fetchall()
        return [i for i in map(self._row, rows) if include_stale or not i.stale]

    def search(
        self, query: str, limit: int = 5, *, include_stale: bool = False
    ) -> list[MemoryItem]:
        return rank(query, self.all(include_stale=include_stale), limit, include_stale)
