"""First-contact smoke test for a live source: one search (limit 1) and one fetch.

Read-only by construction (same allowlisted tools). Reports counts and ids only, never record
content, so its output is safe to paste into an issue.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mcp import Client
from pydantic import BaseModel

from sentinel.guards.redaction import redact_str
from sentinel.mcp_clients.tools import NORMALIZERS, SOURCE_TOOLS, ToolCallError, call_json
from sentinel.retrieval.models import Source


class SmokeReport(BaseModel):
    source: Source
    tools: list[str] = []
    read_only: bool = False
    search_results: int | None = None
    fetched_id: str | None = None
    text_chars: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def describe() -> dict[Source, list[str]]:
    """What the smoke test calls, per source (for printing before asking for approval)."""
    return {
        s: [f"{spec.search_tool}(query='', limit=1)", f"{spec.get_tool}(<first result>)"]
        for s, spec in SOURCE_TOOLS.items()
    }


async def smoke(source: Source, client: Client, timeout_s: float = 30.0) -> SmokeReport:
    report = SmokeReport(source=source)
    spec = SOURCE_TOOLS[source]
    try:
        tools = (await client.list_tools()).tools
        report.tools = sorted(t.name for t in tools)
        report.read_only = all(t.annotations and t.annotations.read_only_hint for t in tools)
        if not report.read_only:
            report.error = "server exposes a non-read-only tool; refusing to continue"
            return report
        found: Any = await call_json(
            client, spec.search_tool, {"query": "", "limit": 1}, timeout_s=timeout_s
        )
        results = found.get("results", [])
        report.search_results = len(results)
        if results:
            record_id = str(results[0][spec.id_field])
            raw = await call_json(
                client, spec.get_tool, {spec.get_arg: record_id}, timeout_s=timeout_s
            )
            doc = NORMALIZERS[source](raw, datetime.now(UTC))
            report.fetched_id = doc.id
            report.text_chars = len(doc.text)
    except (ToolCallError, KeyError, TypeError, ValueError) as exc:
        report.error = redact_str(f"{type(exc).__name__}: {exc}")
    return report
