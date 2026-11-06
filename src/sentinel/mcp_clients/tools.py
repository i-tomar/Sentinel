"""Per-source MCP tool catalog and JSON calls, shared by ingestion and the agent.

`READ_TOOLS` is the allowlist the agent may call in step (f). Write and destructive tools
(update_issue, create_draft, send) are not reachable until the permission gate lands (step l).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from mcp import Client
from mcp.types import TextContent
from pydantic import BaseModel

from sentinel.retrieval.models import Document, Source
from sentinel.retrieval.normalize import gmail_document, jira_document, notion_document


class ToolCallError(RuntimeError):
    pass


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
READ_TOOLS: frozenset[tuple[Source, str]] = frozenset(
    (source, tool)
    for source, spec in SOURCE_TOOLS.items()
    for tool in (spec.search_tool, spec.get_tool)
)
NORMALIZERS: dict[Source, Callable[[Any, datetime], Document]] = {
    "notion": notion_document,
    "jira": jira_document,
    "gmail": gmail_document,
}


async def call_json(
    client: Client, tool: str, args: dict[str, Any], timeout_s: float | None = None
) -> Any:
    """Call an MCP tool that returns JSON text. Tool errors and bad JSON raise ToolCallError."""
    result = await client.call_tool(tool, args, read_timeout_seconds=timeout_s)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    if result.is_error:
        raise ToolCallError(f"{tool} failed: {text}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"{tool} returned malformed JSON") from exc
