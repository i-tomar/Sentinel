"""A tool call to authorize, its fingerprint, and the grant that lets it run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class PermissionDenied(RuntimeError):  # noqa: N818 - reads better at call sites than *Error
    """A tool call was refused before reaching the tool."""


class Action(BaseModel):
    """One concrete tool invocation. Approvals are bound to its fingerprint."""

    model_config = ConfigDict(frozen=True)

    source: str
    tool: str
    args: dict[str, Any]

    def fingerprint(self) -> str:
        """sha256 over source, tool and every argument (canonical JSON, key order ignored)."""
        canonical = json.dumps(
            {"source": self.source, "tool": self.tool, "args": self.args},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ActionRecord(BaseModel):
    """What happened to one proposed action in a turn (summary only, never raw arguments)."""

    approval_id: str | None
    source: str
    tool: str
    risk: Literal["read", "write", "destructive"] | None
    summary: dict[str, Any]
    status: Literal["pending_approval", "policy_denied"]
    reason: str
    confirmation: str | None = None
    """For destructive actions: the exact text to type when approving."""


@dataclass(frozen=True)
class Grant:
    """Proof that an approval was consumed for exactly one action. Only
    `ApprovalStore.consume` hands these out; `call_json` checks the fingerprint again."""

    approval_id: str
    action_hash: str
