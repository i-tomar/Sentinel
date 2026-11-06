"""Memory MCP server: search_memory, get_memory (read); put_memory, mark_stale (write).

Backed by the same MemoryBackend the agent uses (SQLite at MEMORY_DB). Write tools are
annotated as non-read-only so MCP clients and Sentinel's permission gate treat them as writes.
Facts written through MCP are always stored unverified with source "mcp": only Sentinel's
verifier, which checks a fact against its source record, may mark memory verified.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field, ValidationError

from sentinel.memory.backend import MemoryBackend
from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from servers.fixtures.common import READ, WRITE, Limit, to_json

MEMORY_DB_ENV = "MEMORY_DB"
DEFAULT_DB = "./.data/memory.sqlite"


def create_server(backend: MemoryBackend | None = None) -> MCPServer:
    memory = backend or SQLiteMemoryBackend(os.environ.get(MEMORY_DB_ENV, DEFAULT_DB))
    server = MCPServer("sentinel-memory", instructions="Sentinel long-term memory.")

    @server.tool(annotations=READ, structured_output=False)
    async def search_memory(query: str, limit: Limit = 5, include_stale: bool = False) -> str:
        """Search remembered facts; fresh, verified facts rank first. Returns JSON."""
        hits = memory.search(query, limit, include_stale=include_stale)
        return to_json({"results": [h.model_dump(mode="json") for h in hits]})

    @server.tool(annotations=READ, structured_output=False)
    async def get_memory(key: str) -> str:
        """Fetch one fact by key. Returns JSON."""
        item = memory.get(key)
        if item is None:
            raise ToolError(f"memory not found: {key}")
        return to_json(item)

    @server.tool(annotations=WRITE, structured_output=False)
    async def put_memory(
        key: str,
        value: Annotated[str, Field(min_length=1, max_length=500)],
        confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    ) -> str:
        """Store a fact (unverified). WRITE: requires approval."""
        try:
            item = MemoryItem(
                key=key,
                value=value,
                source="mcp",
                verified=False,
                updated_at=datetime.now(UTC),
                confidence=confidence,
            )
        except ValidationError as exc:
            raise ToolError(f"invalid memory item: {exc.error_count()} error(s)") from exc
        return to_json(memory.put(item))

    @server.tool(annotations=WRITE, structured_output=False)
    async def mark_stale(key: str, reason: str, superseded_by: str | None = None) -> str:
        """Flag a fact as stale. WRITE: requires approval."""
        item = memory.mark_stale(key, reason, superseded_by)
        if item is None:
            raise ToolError(f"memory not found: {key}")
        return to_json(item)

    return server


if __name__ == "__main__":
    create_server().run()
