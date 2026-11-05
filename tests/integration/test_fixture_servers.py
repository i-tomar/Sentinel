from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, MCPError, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from servers.fixtures import failure_modes, gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.common import DEFAULT_DATA_DIR

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

SERVERS: dict[str, Callable[[], MCPServer]] = {
    "notion": notion_mcp.create_server,
    "jira": jira_mcp.create_server,
    "gmail": gmail_mcp.create_server,
}
EXPECTED_TOOLS = {
    "notion": {"search_pages": "read", "get_page": "read"},
    "jira": {"search_issues": "read", "get_issue": "read", "update_issue": "write"},
    "gmail": {
        "search_threads": "read",
        "get_thread": "read",
        "create_draft": "write",
        "send": "destructive",
    },
}


@pytest.fixture(autouse=True)
def no_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(failure_modes.FAILURES_ENV, raising=False)


@pytest.fixture
async def notion() -> AsyncIterator[Client]:
    async with Client(notion_mcp.create_server()) as client:
        yield client


@pytest.fixture
async def jira() -> AsyncIterator[Client]:
    async with Client(jira_mcp.create_server()) as client:
        yield client


@pytest.fixture
async def gmail() -> AsyncIterator[Client]:
    async with Client(gmail_mcp.create_server()) as client:
        yield client


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def call(client: Client, tool: str, **args: Any) -> Any:
    result = await client.call_tool(tool, args)
    assert not result.is_error, _text(result)
    return json.loads(_text(result))


async def call_error(client: Client, tool: str, **args: Any) -> str:
    result = await client.call_tool(tool, args)
    assert result.is_error
    return _text(result)


def _risk(read_only: bool | None, destructive: bool | None) -> str:
    if read_only:
        return "read"
    return "destructive" if destructive else "write"


@pytest.mark.parametrize("source", sorted(SERVERS))
async def test_lists_tools_with_risk_annotations(source: str) -> None:
    async with Client(SERVERS[source]()) as client:
        tools = (await client.list_tools()).tools
    found = {
        t.name: _risk(t.annotations.read_only_hint, t.annotations.destructive_hint)
        for t in tools
        if t.annotations is not None
    }
    assert found == EXPECTED_TOOLS[source]
    assert all(t.description for t in tools)


# --- Notion -------------------------------------------------------------------------------


async def test_notion_search_and_get(notion: Client) -> None:
    found = await call(notion, "search_pages", query="auth migration plan", limit=5)
    ids = [r["id"] for r in found["results"]]
    assert {"auth-migration-v1", "auth-migration-v2"} <= set(ids)
    assert ids.index("auth-migration-v2") < ids.index("auth-migration-v1")  # newer first on ties

    page = await call(notion, "get_page", page_id="auth-migration-v2")
    assert page["owner"] == "priya.sharma@acme.test"
    assert page["url"].endswith("/auth-migration-v2")
    assert "PROJ-120" in page["body"]


async def test_notion_unknown_page_is_tool_error(notion: Client) -> None:
    assert "not found" in await call_error(notion, "get_page", page_id="nope")


async def test_notion_no_match_returns_empty(notion: Client) -> None:
    assert await call(notion, "search_pages", query="zzqxv") == {"results": [], "total": 0}


async def test_limit_is_validated(notion: Client) -> None:
    await call_error(notion, "search_pages", query="auth", limit=0)


# --- Jira ---------------------------------------------------------------------------------


async def test_jira_blocked_by_filter(jira: Client) -> None:
    found = await call(jira, "search_issues", blocked_by="PROJ-120")
    assert {r["key"] for r in found["results"]} == {"PROJ-142", "PROJ-147", "PROJ-151"}
    assert all(r["status"] == "Blocked" for r in found["results"])


async def test_jira_exact_key_search_ranks_first(jira: Client) -> None:
    found = await call(jira, "search_issues", query="PROJ-142 billing login")
    assert found["results"][0]["key"] == "PROJ-142"


async def test_jira_get_issue_includes_comments(jira: Client) -> None:
    issue = await call(jira, "get_issue", key="proj-155")
    assert issue["key"] == "PROJ-155"
    assert len(issue["comments"]) == 2


async def test_jira_update_is_in_memory_only(jira: Client) -> None:
    before = (DEFAULT_DATA_DIR / "jira_issues.json").read_bytes()
    updated = await call(jira, "update_issue", key="PROJ-142", status="Done", comment="test")
    assert updated["status"] == "Done"
    assert updated["comments"][-1]["author"] == "sentinel-agent@acme.test"
    assert (await call(jira, "get_issue", key="PROJ-142"))["status"] == "Done"

    async with Client(jira_mcp.create_server()) as fresh:
        assert (await call(fresh, "get_issue", key="PROJ-142"))["status"] == "Blocked"
    assert (DEFAULT_DATA_DIR / "jira_issues.json").read_bytes() == before


async def test_jira_update_requires_a_change(jira: Client) -> None:
    assert "nothing to update" in await call_error(jira, "update_issue", key="PROJ-142")


async def test_jira_rejects_invalid_status(jira: Client) -> None:
    await call_error(jira, "update_issue", key="PROJ-142", status="Shipped")


# --- Gmail --------------------------------------------------------------------------------


async def test_gmail_last_week_escalations(gmail: Client) -> None:
    found = await call(
        gmail, "search_threads", label="escalation", after="2026-10-26", before="2026-11-02"
    )
    assert {r["id"] for r in found["results"]} == {"thr-0003", "thr-0007"}


async def test_gmail_search_by_ticket_and_get_thread(gmail: Client) -> None:
    found = await call(gmail, "search_threads", query="PROJ-147")
    assert "thr-0003" in {r["id"] for r in found["results"]}
    thread = await call(gmail, "get_thread", thread_id="thr-0003")
    assert thread["messages"][0]["sender"] == "dana.whitfield@globex.test"


async def test_gmail_draft_then_send_stays_in_process(gmail: Client) -> None:
    draft = await call(
        gmail,
        "create_draft",
        to=["Dana.Whitfield@globex.test"],
        subject="Update",
        body="hi",
        thread_id="thr-0003",
    )
    assert draft["draft_id"] == "draft-0001"
    assert draft["to"] == ["dana.whitfield@globex.test"]
    sent = await call(gmail, "send", draft_id=draft["draft_id"])
    assert sent["message_id"] == "msg-0001"
    assert "not found" in await call_error(gmail, "send", draft_id=draft["draft_id"])


async def test_gmail_draft_validation(gmail: Client) -> None:
    assert "recipient" in await call_error(gmail, "create_draft", to=[], subject="s", body="b")
    assert "not found" in await call_error(
        gmail, "create_draft", to=["a@acme.test"], subject="s", body="b", thread_id="thr-9999"
    )


# --- Failure modes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"), [("error_500", "500 Internal"), ("rate_limit", "429 Too Many")]
)
async def test_error_modes_surface_as_tool_errors(
    monkeypatch: pytest.MonkeyPatch, gmail: Client, mode: str, expected: str
) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, f"gmail={mode}")
    assert expected in await call_error(gmail, "search_threads", query="invoice")


async def test_empty_and_malformed_modes(
    monkeypatch: pytest.MonkeyPatch, jira: Client, notion: Client
) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "jira=empty,notion.get_page=malformed")
    assert await call(jira, "search_issues", blocked_by="PROJ-120") == {"results": [], "total": 0}
    result = await notion.call_tool("get_page", {"page_id": "auth-migration-v2"})
    assert not result.is_error
    with pytest.raises(json.JSONDecodeError):
        json.loads(_text(result))
    assert (await call(notion, "search_pages", query="auth"))["total"] > 0  # other tool unaffected


async def test_timeout_mode_trips_client_read_timeout(
    monkeypatch: pytest.MonkeyPatch, notion: Client
) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "notion=timeout")
    monkeypatch.setenv(failure_modes.TIMEOUT_ENV, "5")
    with pytest.raises(MCPError):
        await notion.call_tool("search_pages", {"query": "auth"}, read_timeout_seconds=0.3)


# --- Real subprocess over stdio -----------------------------------------------------------


def _stdio(source: str, env: dict[str, str] | None = None) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", f"servers.fixtures.{source}_mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


@pytest.mark.parametrize("source", sorted(SERVERS))
async def test_stdio_server_starts_and_lists_tools(source: str) -> None:
    async with Client(_stdio(source)) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == set(EXPECTED_TOOLS[source])


async def test_stdio_server_honours_failure_env() -> None:
    env = {failure_modes.FAILURES_ENV: "jira.get_issue=rate_limit"}
    async with Client(_stdio("jira", env)) as client:
        assert "429" in await call_error(client, "get_issue", key="PROJ-142")
        assert (await call(client, "search_issues", query="OIDC"))["total"] > 0
