"""Trusted tool registry: every tool Sentinel may call, its risk class, argument schema, and
an audit-safe summary of its arguments.

This table, not tool descriptions, server annotations, model output or retrieved text, decides
what a tool may do. A tool that is not listed here is denied. Tests check that each fixture and
memory server's own annotations agree with it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.memory.models import MEMORY_KEY_PATTERN, MemoryItem

Risk = Literal["read", "write", "destructive"]
Summary = dict[str, Any]

TICKET_KEY = r"^[A-Z][A-Z0-9]{1,9}-\d{1,6}$"
EMAIL = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
MAX_RECIPIENTS = 10


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpdateIssueArgs(_Args):
    key: str = Field(pattern=TICKET_KEY)
    status: Literal["To Do", "In Progress", "Blocked", "In Review", "Done"] | None = None
    assignee: str | None = Field(default=None, min_length=1, max_length=120)
    comment: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _something_to_change(self) -> UpdateIssueArgs:
        if self.status is None and self.assignee is None and self.comment is None:
            raise ValueError("pass status, assignee and/or comment")
        return self


class CreateDraftArgs(_Args):
    to: list[str] = Field(min_length=1, max_length=MAX_RECIPIENTS)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=5000)
    thread_id: str | None = Field(default=None, pattern=r"^thr-\d{4}$")

    @model_validator(mode="after")
    def _addresses(self) -> CreateDraftArgs:
        bad = [a for a in self.to if not re.fullmatch(EMAIL, a)]
        if bad:
            raise ValueError(f"{len(bad)} invalid recipient address(es)")
        return self


class SendArgs(_Args):
    draft_id: str = Field(pattern=r"^draft-\d{4}$")


class MarkStaleArgs(_Args):
    key: str = Field(pattern=MEMORY_KEY_PATTERN)
    reason: str = Field(min_length=1, max_length=300)
    superseded_by: str | None = Field(default=None, max_length=200)


def _domains(addresses: list[str]) -> list[str]:
    return sorted({a.rsplit("@", 1)[-1].lower() for a in addresses})


def _update_issue(args: Mapping[str, Any]) -> Summary:
    return {
        "key": args.get("key"),
        "status": args.get("status"),
        "assignee_set": args.get("assignee") is not None,
        "comment_chars": len(str(args.get("comment") or "")),
    }


def _create_draft(args: Mapping[str, Any]) -> Summary:
    to = [str(a) for a in args.get("to") or []]
    return {
        "recipients": len(to),
        "recipient_domains": _domains(to),
        "subject_chars": len(str(args.get("subject") or "")),
        "body_chars": len(str(args.get("body") or "")),
        "thread_id": args.get("thread_id"),
    }


def _send(args: Mapping[str, Any]) -> Summary:
    return {"draft_id": args.get("draft_id")}


def _key_hash(key: Any) -> str:
    """Memory keys are built from the words of a question, so only a short hash is logged."""
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def _put_memory(args: Mapping[str, Any]) -> Summary:
    return {
        "key_sha256": _key_hash(args.get("key")),
        "source": args.get("source"),
        "verified": args.get("verified"),
        "value_chars": len(str(args.get("value") or "")),
    }


def _mark_stale(args: Mapping[str, Any]) -> Summary:
    return {
        "key_sha256": _key_hash(args.get("key")),
        "superseded_by": args.get("superseded_by"),
        "reason_chars": len(str(args.get("reason") or "")),
    }


def _names(args: Mapping[str, Any]) -> Summary:
    return {"arg_names": sorted(args)}


@dataclass(frozen=True)
class ToolSpec:
    source: str
    name: str
    risk: Risk
    args: type[BaseModel] | None = None
    """Argument schema, enforced for write and destructive tools."""
    summarize: Callable[[Mapping[str, Any]], Summary] = _names
    """What the audit log may keep about the arguments (no free text, no addresses)."""


_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("notion", "search_pages", "read"),
    ToolSpec("notion", "get_page", "read"),
    ToolSpec("jira", "search_issues", "read"),
    ToolSpec("jira", "get_issue", "read"),
    ToolSpec("jira", "update_issue", "write", UpdateIssueArgs, _update_issue),
    ToolSpec("gmail", "search_threads", "read"),
    ToolSpec("gmail", "get_thread", "read"),
    ToolSpec("gmail", "create_draft", "write", CreateDraftArgs, _create_draft),
    ToolSpec("gmail", "send", "destructive", SendArgs, _send),
    ToolSpec("memory", "search_memory", "read"),
    ToolSpec("memory", "get_memory", "read"),
    ToolSpec("memory", "put_memory", "write", MemoryItem, _put_memory),
    ToolSpec("memory", "mark_stale", "write", MarkStaleArgs, _mark_stale),
)
REGISTRY: dict[tuple[str, str], ToolSpec] = {(s.source, s.name): s for s in _SPECS}
_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in _SPECS}
if len(_BY_NAME) != len(_SPECS):  # call_json resolves tools by name alone
    raise RuntimeError("tool names in the registry must be unique")

READ_TOOLS: frozenset[tuple[str, str]] = frozenset(
    key for key, spec in REGISTRY.items() if spec.risk == "read" and spec.source != "memory"
)
"""Source tools the agent's read path may call without approval."""


def lookup(source: str, tool: str) -> ToolSpec | None:
    return REGISTRY.get((source, tool))


def spec_by_name(tool: str) -> ToolSpec | None:
    return _BY_NAME.get(tool)


def summarize(source: str, tool: str, args: Mapping[str, Any]) -> Summary:
    spec = lookup(source, tool)
    return (spec.summarize if spec else _names)(args)
