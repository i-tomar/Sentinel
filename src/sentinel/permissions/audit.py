"""Append-only JSONL audit trail of proposed and executed actions.

Each record carries identifiers (session, user, approval id, sha256 of the action) and a
per-tool summary from the registry: ticket keys, statuses, counts and lengths, recipient
domains. It never stores the user's question, retrieved text, credentials, email addresses,
message bodies, memory values, or raw tool error messages. `reason` is a short fixed phrase.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from sentinel.guards.redaction import redact_str
from sentinel.mcp_clients.registry import lookup, summarize
from sentinel.permissions.actions import Action

AuditEvent = Literal[
    "proposed",  # approval requested for an action
    "policy_denied",  # refused by policy without asking
    "approved",
    "denied",
    "attempted",  # grant consumed, call starting
    "succeeded",
    "failed",  # the tool clearly did not apply the write (rejected, rate limited, ...)
    "uncertain",  # sent, but the outcome is unknown (timeout, crash, 5xx, unreadable reply)
    "blocked",  # execution refused: missing, denied, expired, used, mismatched approval
]


class AuditRecord(BaseModel):
    ts: datetime
    event: AuditEvent
    session: str
    principal: str
    approval_id: str | None
    source: str
    tool: str
    risk: str | None
    action_hash: str
    args: dict[str, Any]
    reason: str | None = None
    result: dict[str, Any] | None = None


def _scrub(values: dict[str, Any]) -> dict[str, Any]:
    """Second line of defence: summaries hold ids and counts, but run strings through the PII
    redactor anyway in case a tool summary ever carries free text."""
    return {k: redact_str(v) if isinstance(v, str) else v for k, v in values.items()}


class AuditLog:
    """Writes to `path` when given; always keeps the records of this process in `records`."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[AuditRecord] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event: AuditEvent,
        action: Action,
        *,
        session: str,
        principal: str,
        approval_id: str | None = None,
        reason: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> AuditRecord:
        spec = lookup(action.source, action.tool)
        entry = AuditRecord(
            ts=datetime.now(UTC),
            event=event,
            session=session,
            principal=principal,
            approval_id=approval_id,
            source=action.source,
            tool=action.tool,
            risk=spec.risk if spec else None,
            action_hash=action.fingerprint(),
            args=_scrub(summarize(action.source, action.tool, action.args)),
            reason=redact_str(reason) if reason else reason,
            result=_scrub(result) if result else result,
        )
        self.records.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        return entry
