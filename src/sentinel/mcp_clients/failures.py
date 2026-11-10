"""What can go wrong with a tool call, what is worth retrying, and how it is described.

Server error text is untrusted: it can echo arguments, record content or an attacker's
instructions. It is only used here to pick a kind. What gets stored, prompted, traced or shown
is the fixed description of that kind, never the server's text.
"""

from __future__ import annotations

import re
from typing import Literal

FailureKind = Literal[
    "timeout",  # no answer in time (client side, or the server says its upstream timed out)
    "rate_limited",  # 429 / too many requests
    "server_error",  # 5xx
    "unavailable",  # not connected: failed to start, crashed, connection closed
    "invalid_response",  # malformed JSON or data that does not fit the expected schema
    "invalid_arguments",  # the server rejected the arguments
    "not_found",
    "tool_error",  # any other error the tool reported
    "permission_denied",  # refused before the call (permission gate)
    "skipped",  # not attempted: the source already failed earlier in this turn
    "budget_exhausted",  # not attempted: the turn's tool time budget ran out
]

RETRYABLE: frozenset[str] = frozenset({"timeout", "rate_limited", "server_error"})
"""Transient errors: worth another attempt with backoff (reads only; writes never retry)."""
SOURCE_DOWN: frozenset[str] = RETRYABLE | {"unavailable"}
"""After one of these (retries exhausted), the source is skipped for the rest of the turn."""
UNCERTAIN_WRITE: frozenset[str] = frozenset(
    {"timeout", "unavailable", "server_error", "invalid_response"}
)
"""A write that failed this way may still have been applied: its outcome is unknown."""

_DESCRIPTIONS: dict[str, str] = {
    "timeout": "timed out",
    "rate_limited": "rate limited (HTTP 429)",
    "server_error": "server error (HTTP 5xx)",
    "unavailable": "unavailable (failed to start, crashed or disconnected)",
    "invalid_response": "invalid response (malformed or unexpected data)",
    "invalid_arguments": "rejected the request arguments",
    "not_found": "record not found",
    "tool_error": "tool error",
    "permission_denied": "tool not allowed",
    "skipped": "skipped: the source failed earlier in this turn",
    "budget_exhausted": "skipped: the turn's time budget for tool calls ran out",
}

_RATE = re.compile(r"\b429\b|too many requests|rate.?limit", re.I)
_SERVER = re.compile(r"\b5\d\d\b|internal server error|service unavailable|bad gateway", re.I)
_TIMEOUT = re.compile(r"timed? ?out|timeout", re.I)
_ARGS = re.compile(r"validation error|invalid (?:argument|param)|missing required", re.I)
_NOT_FOUND = re.compile(r"\bnot found\b|\b404\b", re.I)


def classify_error(text: str) -> FailureKind:
    """Kind of an error a tool reported (`is_error`), judged from its text only.

    Argument and not-found errors are checked first: their text echoes the request (an id like
    PROJ-500, an input value), which must not make a permanent error look transient."""
    if _ARGS.search(text):
        return "invalid_arguments"
    if _NOT_FOUND.search(text):
        return "not_found"
    if _RATE.search(text):
        return "rate_limited"
    if _TIMEOUT.search(text):
        return "timeout"
    if _SERVER.search(text):
        return "server_error"
    return "tool_error"


def describe(kind: str, timeout_s: float | None = None) -> str:
    if kind == "timeout" and timeout_s is not None:
        return f"timed out after {timeout_s:g}s"
    return _DESCRIPTIONS.get(kind, "tool error")
