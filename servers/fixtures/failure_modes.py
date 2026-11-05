"""Env-driven failure injection for the fixture MCP servers.

`SENTINEL_FIXTURE_FAILURES` holds comma-separated `target=mode` rules, read on every call so a
harness can change behaviour per case without restarting servers:

    SENTINEL_FIXTURE_FAILURES="gmail=timeout,jira.get_issue=error_500,*=rate_limit"

Targets are `<source>.<tool>`, `<source>`, or `*`; the most specific match wins.
`SENTINEL_FIXTURE_TIMEOUT_S` (default 30) is how long `timeout` stalls before giving up.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, TypeGuard, get_args

import anyio
from mcp.server.mcpserver.exceptions import ToolError

FailureMode = Literal["timeout", "error_500", "empty", "malformed", "rate_limit"]
FAILURE_MODES: frozenset[str] = frozenset(get_args(FailureMode))

FAILURES_ENV = "SENTINEL_FIXTURE_FAILURES"
TIMEOUT_ENV = "SENTINEL_FIXTURE_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 30.0
MALFORMED_PAYLOAD = '{"results": [{"id": "trunc'


def parse_spec(spec: str) -> dict[str, FailureMode]:
    rules: dict[str, FailureMode] = {}
    for raw in spec.split(","):
        rule = raw.strip()
        if not rule:
            continue
        target, sep, mode = (part.strip() for part in rule.partition("="))
        if not sep or not target:
            raise ValueError(f"invalid failure rule {rule!r}; expected target=mode")
        if not _is_mode(mode):
            expected = ", ".join(sorted(FAILURE_MODES))
            raise ValueError(f"unknown failure mode {mode!r}; expected one of: {expected}")
        rules[target] = mode
    return rules


def _is_mode(value: str) -> TypeGuard[FailureMode]:
    return value in FAILURE_MODES


def mode_for(source: str, tool: str, env: Mapping[str, str] | None = None) -> FailureMode | None:
    env = os.environ if env is None else env
    rules = parse_spec(env.get(FAILURES_ENV, ""))
    return rules.get(f"{source}.{tool}") or rules.get(source) or rules.get("*")


def timeout_seconds(env: Mapping[str, str] | None = None) -> float:
    env = os.environ if env is None else env
    return float(env.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))


async def apply(source: str, tool: str, *, empty: str) -> str | None:
    """Return an override payload, raise a ToolError, or return None to run the tool normally.

    `empty` is the tool's well-formed "no results" payload, returned in `empty` mode.
    """
    mode = mode_for(source, tool)
    if mode is None:
        return None
    if mode == "timeout":
        seconds = timeout_seconds()
        await anyio.sleep(seconds)
        raise ToolError(f"{source} upstream timed out after {seconds:g}s")
    if mode == "error_500":
        raise ToolError(f"500 Internal Server Error from {source} API")
    if mode == "rate_limit":
        raise ToolError(f"429 Too Many Requests from {source} API; retry after 1s")
    if mode == "empty":
        return empty
    return MALFORMED_PAYLOAD
