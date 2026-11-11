"""Per-source MCP tool catalog and JSON calls, shared by ingestion, the agent and approvals.

`call_json` is the one place Sentinel calls an MCP tool, so the permission check lives here:
tools missing from the trusted registry are refused, and write/destructive tools run only with
a `Grant` (a consumed approval) for exactly these arguments. Reads need nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from mcp import Client, MCPError
from mcp.types import REQUEST_TIMEOUT, TextContent
from pydantic import BaseModel

from sentinel.mcp_clients.failures import RETRYABLE as RETRYABLE  # re-exported
from sentinel.mcp_clients.failures import FailureKind
from sentinel.mcp_clients.failures import classify_error as classify_error  # re-exported
from sentinel.mcp_clients.registry import READ_TOOLS as READ_TOOLS  # re-exported
from sentinel.mcp_clients.registry import spec_by_name
from sentinel.permissions.actions import Action, Grant, PermissionDenied
from sentinel.retrieval.models import Document, Source
from sentinel.retrieval.normalize import gmail_document, jira_document, notion_document


class ToolCallError(RuntimeError):
    """A tool call that did not produce usable JSON. `kind` is a `FailureKind`; the message
    may contain server text and must never be shown, stored or prompted (use `describe`)."""

    def __init__(self, message: str, kind: FailureKind = "tool_error") -> None:
        super().__init__(message)
        self.kind: FailureKind = kind


class SourceTools(BaseModel):
    search_tool: str
    get_tool: str
    get_arg: str
    id_field: str


SOURCE_TOOLS: dict[Source, SourceTools] = {
    "notion": SourceTools(
        search_tool="search_pages", get_tool="get_page", get_arg="page_id", id_field="id"
    ),
    "jira": SourceTools(
        search_tool="search_issues", get_tool="get_issue", get_arg="key", id_field="key"
    ),
    "gmail": SourceTools(
        search_tool="search_threads", get_tool="get_thread", get_arg="thread_id", id_field="id"
    ),
}
NORMALIZERS: dict[Source, Callable[[Any, datetime], Document]] = {
    "notion": notion_document,
    "jira": jira_document,
    "gmail": gmail_document,
}


def authorize(tool: str, args: dict[str, Any], grant: Grant | None) -> None:
    """Raise PermissionDenied unless `tool` is a registered read, or `grant` covers exactly
    this write. Runs before any byte reaches the server."""
    spec = spec_by_name(tool)
    if spec is None:
        raise PermissionDenied(f"unknown tool {tool!r}: not in the trusted registry")
    if spec.risk == "read":
        return
    if grant is None:
        raise PermissionDenied(f"{spec.source}.{tool} is a {spec.risk} tool and needs an approval")
    if grant.action_hash != Action(source=spec.source, tool=tool, args=args).fingerprint():
        raise PermissionDenied(f"approval {grant.approval_id} does not match this call")


async def call_json(
    client: Client,
    tool: str,
    args: dict[str, Any],
    timeout_s: float | None = None,
    *,
    grant: Grant | None = None,
) -> Any:
    """Call an MCP tool that returns JSON text. Calls the permission check refuses raise
    PermissionDenied; everything else that goes wrong (tool errors, bad JSON, timeouts, a
    closed or crashed connection) raises a classified ToolCallError, never something else."""
    authorize(tool, args, grant)
    try:
        result = await client.call_tool(tool, args, read_timeout_seconds=timeout_s)
    except MCPError as exc:
        kind: FailureKind = "timeout" if exc.code == REQUEST_TIMEOUT else "unavailable"
        raise ToolCallError(f"{tool}: {type(exc).__name__}", kind=kind) from None
    except Exception as exc:
        raise ToolCallError(f"{tool}: {type(exc).__name__}", kind="unavailable") from None
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    if result.is_error:
        raise ToolCallError(f"{tool} failed: {text}", kind=classify_error(text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ToolCallError(f"{tool} returned malformed JSON", kind="invalid_response") from None
