"""Approval records in SQLite: one per proposed write, bound to session, user and fingerprint.

Lifecycle: pending -> approved | denied | expired; approved -> consumed | expired. Only an
approved, unexpired record whose session, user and argument fingerprint all match can be
consumed, and only once: a conditional UPDATE (`... WHERE status = 'approved'`) decides the
winner even between processes, and consumption happens before the write is dispatched, so a
write runs at most once per approval. Arguments are kept only while the record is open, because
the approved action has to be replayed exactly. They are cleared when it is consumed or denied,
and overdue requests are expired (arguments cleared) every time a store is opened or a new
request is made, whether or not anyone looks at them again. There is no background timer.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from sentinel.mcp_clients.registry import Risk
from sentinel.permissions.actions import Action, Grant

Status = Literal["pending", "approved", "denied", "consumed", "expired"]
DEFAULT_TTL_S = 900.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    principal TEXT NOT NULL,
    source TEXT NOT NULL,
    tool TEXT NOT NULL,
    risk TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    args_json TEXT,
    confirmation TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_at TEXT,
    decided_via TEXT
)
"""
_COLUMNS = (
    "id, session, principal, source, tool, risk, action_hash, args_json, confirmation, status, "
    "created_at, expires_at, decided_at, decided_via"
)


class ApprovalError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Approval(BaseModel):
    id: str
    session: str
    principal: str
    source: str
    tool: str
    risk: Risk
    action_hash: str
    args: dict[str, Any] | None
    confirmation: str | None
    status: Status
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None
    decided_via: str | None

    def action(self) -> Action | None:
        """The approved action, while its arguments are still held."""
        if self.args is None:
            return None
        return Action(source=self.source, tool=self.tool, args=self.args)


def _now() -> datetime:
    return datetime.now(UTC)


class ApprovalStore:
    def __init__(
        self,
        path: Path | str,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=10.0
        )
        # SQLite overwrites deleted/cleared content with zeros instead of leaving it in free
        # pages. This is hygiene, not secure erasure (see the README's retention notes).
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.execute(_SCHEMA)
        self._lock = threading.Lock()
        self.ttl = timedelta(seconds=ttl_s)
        self.clock = clock
        self.expire_overdue()

    def expire_overdue(self) -> int:
        """Close every open approval past its deadline and drop its arguments."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE approvals SET status = 'expired', args_json = NULL, decided_at = ? "
                "WHERE status IN ('pending', 'approved') AND expires_at <= ?",
                (self.clock().isoformat(), self.clock().isoformat()),
            )
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()

    # --- reads -------------------------------------------------------------------------

    def get(self, approval_id: str) -> Approval | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM approvals WHERE id = ?",  # noqa: S608 - constant columns
            (approval_id,),
        ).fetchone()
        return _row(row) if row else None

    def list(self, session: str | None = None) -> list[Approval]:
        sql = f"SELECT {_COLUMNS} FROM approvals"  # noqa: S608 - constant columns
        rows = (
            self._conn.execute(f"{sql} WHERE session = ? ORDER BY created_at", (session,))
            if session is not None
            else self._conn.execute(f"{sql} ORDER BY created_at")
        ).fetchall()
        return [_row(r) for r in rows]

    # --- transitions -------------------------------------------------------------------

    def request(
        self,
        action: Action,
        *,
        risk: Risk,
        session: str,
        principal: str,
        confirmation: str | None = None,
    ) -> Approval:
        self.expire_overdue()
        now = self.clock()
        approval_id = f"apr-{secrets.token_hex(8)}"
        with self._lock:
            self._conn.execute(
                f"INSERT INTO approvals ({_COLUMNS}) VALUES "  # noqa: S608 - constant columns
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)",
                (
                    approval_id,
                    session,
                    principal,
                    action.source,
                    action.tool,
                    risk,
                    action.fingerprint(),
                    json.dumps(action.args, sort_keys=True, default=str),
                    confirmation,
                    now.isoformat(),
                    (now + self.ttl).isoformat(),
                ),
            )
        return self._must(approval_id)

    def approve(
        self,
        approval_id: str,
        *,
        session: str,
        principal: str,
        confirmation: str | None = None,
        via: str = "user",
    ) -> Approval:
        with self._lock:
            approval = self._open(approval_id, session, principal, expect="pending")
            if approval.confirmation is not None and confirmation != approval.confirmation:
                raise ApprovalError(
                    f"typed confirmation required: type exactly {approval.confirmation!r}"
                )
            self._set(approval_id, "approved", via, keep_args=True)
        return self._must(approval_id)

    def deny(self, approval_id: str, *, session: str, principal: str) -> Approval:
        with self._lock:
            self._open(approval_id, session, principal, expect="pending")
            self._set(approval_id, "denied", "user", keep_args=False)
        return self._must(approval_id)

    def consume(self, approval_id: str, action: Action, *, session: str, principal: str) -> Grant:
        """Turn an approval into a single-use grant for exactly `action`."""
        with self._lock:
            approval = self._open(approval_id, session, principal, expect="approved")
            if (approval.source, approval.tool) != (action.source, action.tool) or (
                approval.action_hash != action.fingerprint()
            ):
                raise ApprovalError("arguments changed since approval: request a new approval")
            cursor = self._conn.execute(
                "UPDATE approvals SET status = 'consumed', args_json = NULL WHERE id = ? "
                "AND status = 'approved'",
                (approval_id,),
            )
            if cursor.rowcount != 1:  # another process got there first
                raise ApprovalError("approval already used")
        return Grant(approval_id=approval_id, action_hash=approval.action_hash)

    # --- helpers -----------------------------------------------------------------------

    def _open(self, approval_id: str, session: str, principal: str, *, expect: Status) -> Approval:
        approval = self.get(approval_id)
        if approval is None:
            raise ApprovalError("no such approval")
        if approval.session != session:
            raise ApprovalError("approval belongs to another session")
        if approval.principal != principal:
            raise ApprovalError("approval belongs to another user")
        closed = {
            "denied": "approval was denied",
            "consumed": "approval already used",
            "expired": "approval expired",
        }
        if approval.status in closed:
            raise ApprovalError(closed[approval.status])
        if self.clock() >= approval.expires_at:
            self._set(approval_id, "expired", None, keep_args=False)
            raise ApprovalError("approval expired")
        if approval.status != expect:
            raise ApprovalError(
                "action not approved yet" if expect == "approved" else "already decided"
            )
        return approval

    def _set(self, approval_id: str, status: Status, via: str | None, *, keep_args: bool) -> None:
        clear = "" if keep_args else ", args_json = NULL"
        sql = (
            "UPDATE approvals SET status = ?, decided_at = ?, "  # noqa: S608 - constant
            f"decided_via = COALESCE(?, decided_via){clear} WHERE id = ?"
        )
        self._conn.execute(sql, (status, self.clock().isoformat(), via, approval_id))

    def _must(self, approval_id: str) -> Approval:
        approval = self.get(approval_id)
        if approval is None:
            raise ApprovalError("no such approval")
        return approval


def _row(row: tuple[Any, ...]) -> Approval:
    (
        id_,
        session,
        principal,
        source,
        tool,
        risk,
        action_hash,
        args_json,
        confirmation,
        status,
        created_at,
        expires_at,
        decided_at,
        decided_via,
    ) = row
    return Approval(
        id=id_,
        session=session,
        principal=principal,
        source=source,
        tool=tool,
        risk=risk,
        action_hash=action_hash,
        args=json.loads(args_json) if args_json else None,
        confirmation=confirmation,
        status=status,
        created_at=datetime.fromisoformat(created_at),
        expires_at=datetime.fromisoformat(expires_at),
        decided_at=datetime.fromisoformat(decided_at) if decided_at else None,
        decided_via=decided_via,
    )
