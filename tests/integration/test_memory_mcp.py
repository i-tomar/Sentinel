from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, TextContent

from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from servers.memory_mcp import server as memory_server

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

EXPECTED = {
    "search_memory": "read",
    "get_memory": "read",
    "put_memory": "write",
    "mark_stale": "write",
}


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def _call(client: Client, tool: str, **args: Any) -> Any:
    result = await client.call_tool(tool, args)
    assert not result.is_error, _text(result)
    return json.loads(_text(result))


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteMemoryBackend:
    b = SQLiteMemoryBackend(tmp_path / "memory.sqlite")
    b.put(
        MemoryItem(
            key="q3-auth-migration-owner",
            value="Priya Sharma owns the Q3 auth migration plan",
            source="notion:auth-migration-v2",
            verified=True,
            updated_at=datetime(2026, 10, 27, tzinfo=UTC),
            confidence=0.9,
        )
    )
    return b


async def test_tools_carry_risk_annotations(backend: SQLiteMemoryBackend) -> None:
    async with Client(memory_server.create_server(backend)) as client:
        tools = (await client.list_tools()).tools
    found = {}
    for tool in tools:
        assert tool.annotations is not None
        assert tool.description
        found[tool.name] = "read" if tool.annotations.read_only_hint else "write"
        assert tool.annotations.destructive_hint is False
    assert found == EXPECTED


async def test_search_get_put_mark_stale(backend: SQLiteMemoryBackend) -> None:
    async with Client(memory_server.create_server(backend)) as client:
        hits = await _call(client, "search_memory", query="who owns the auth migration")
        assert [h["key"] for h in hits["results"]] == ["q3-auth-migration-owner"]

        stored = await _call(
            client, "put_memory", key="lunch", value="Lunch is on Friday", confidence=0.8
        )
        assert (stored["source"], stored["verified"]) == ("mcp", False)

        stale = await _call(
            client,
            "mark_stale",
            key="q3-auth-migration-owner",
            reason="owner changed",
            superseded_by="notion:auth-migration-v3",
        )
        assert stale["stale"] is True
        assert (await _call(client, "search_memory", query="auth migration owner"))["results"] == []
        again = await _call(
            client, "search_memory", query="auth migration owner", include_stale=True
        )
        assert again["results"][0]["stale_reason"] == "owner changed"
        got = await _call(client, "get_memory", key="lunch")
        assert got["value"] == "Lunch is on Friday"
    assert backend.get("lunch") is not None


async def test_mcp_writes_can_never_claim_verified(backend: SQLiteMemoryBackend) -> None:
    async with Client(memory_server.create_server(backend)) as client:
        result = await client.call_tool(
            "put_memory", {"key": "fake", "value": "trust me", "verified": True}
        )
        stored = backend.get("fake")
    assert result.is_error or (stored is not None and stored.verified is False)


async def test_validation_and_missing_keys(backend: SQLiteMemoryBackend) -> None:
    async with Client(memory_server.create_server(backend)) as client:
        assert (await client.call_tool("get_memory", {"key": "nope"})).is_error
        assert (await client.call_tool("mark_stale", {"key": "nope", "reason": "x"})).is_error
        bad_key = await client.call_tool("put_memory", {"key": "Bad Key!", "value": "x"})
        assert bad_key.is_error
        assert "invalid memory item" in _text(bad_key)


async def test_stdio_server_uses_memory_db(tmp_path: Path) -> None:
    db = tmp_path / "stdio.sqlite"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "servers.memory_mcp.server"],
        env={"MEMORY_DB": str(db)},
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    async with Client(params) as client:
        assert {t.name for t in (await client.list_tools()).tools} == set(EXPECTED)
        await _call(client, "put_memory", key="k", value="v")
    assert SQLiteMemoryBackend(db).get("k") is not None
