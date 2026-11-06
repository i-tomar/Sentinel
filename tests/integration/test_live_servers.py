"""Live MCP servers end to end over mocked HTTP: tool surface, ingestion, agent, smoke test,
Gmail consent. No real account is contacted (the network guard would fail the test)."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from mcp import Client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from sentinel.config import Settings
from sentinel.graph.build import Deps, ask_async
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.live.gmail import READONLY_SCOPE, GmailAuth, GmailClient
from sentinel.live.gmail_consent import obtain_refresh_token
from sentinel.live.http import LiveAPIError
from sentinel.live.jira import JiraClient
from sentinel.live.notion import API as NOTION_API
from sentinel.live.notion import NotionClient
from sentinel.live.smoke import smoke
from sentinel.llm import MockLLM
from sentinel.mcp_clients.tools import READ_TOOLS
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.ingest import ingest
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import build_tracer_provider, get_tracer
from servers.live import gmail_mcp, jira_mcp, notion_mcp
from tests.live_payloads import (
    PAGE_V1,
    PAGE_V2,
    THREAD_ID,
    Recorder,
    gmail_thread,
    jira_issue,
    notion_blocks,
    notion_page,
)

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _notion(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == f"{NOTION_API}/search":
        return httpx.Response(
            200,
            json={
                "results": [
                    notion_page(PAGE_V2, "Q3 Auth Migration Plan (v2)", "2026-10-27T00:00:00.000Z"),
                    notion_page(PAGE_V1, "Q3 Auth Migration Plan (v1)", "2026-07-05T00:00:00.000Z"),
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )
    for page_id, title, edited, owner in (
        (PAGE_V2, "Q3 Auth Migration Plan (v2)", "2026-10-27T00:00:00.000Z", "Priya Sharma"),
        (PAGE_V1, "Q3 Auth Migration Plan (v1)", "2026-07-05T00:00:00.000Z", "Vikram Rao"),
    ):
        if url == f"{NOTION_API}/pages/{page_id}":
            return httpx.Response(200, json=notion_page(page_id, title, edited))
        if url.startswith(f"{NOTION_API}/blocks/{page_id}/children"):
            return httpx.Response(
                200,
                json=notion_blocks(("paragraph", f"Owner of the auth migration plan: {owner}.")),
            )
    return httpx.Response(404, json={"code": "object_not_found"})


def _jira(request: httpx.Request) -> httpx.Response:
    path = urlparse(str(request.url)).path
    if path == "/rest/api/3/search/jql":
        return httpx.Response(
            200,
            json={
                "issues": [
                    jira_issue("PROJ-142", "Billing login", "Blocked", blocked_by="PROJ-120")
                ],
                "isLast": True,
            },
        )
    if path == "/rest/api/3/issue/PROJ-142":
        return httpx.Response(
            200, json=jira_issue("PROJ-142", "Billing login", "Blocked", blocked_by="PROJ-120")
        )
    return httpx.Response(404, json={})


def _token(scope: str = READONLY_SCOPE) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "access_token": "at-fresh",
                "refresh_token": "rt-new",
                "scope": scope,
                "expires_in": 3600,
            },
        )
    )


def _gmail(request: httpx.Request) -> httpx.Response:
    path = urlparse(str(request.url)).path
    if path.endswith("/threads"):
        return httpx.Response(200, json={"threads": [{"id": THREAD_ID, "snippet": "x"}]})
    return httpx.Response(200, json=gmail_thread())


@pytest.fixture
async def live_clients() -> AsyncIterator[dict[Source, Client]]:
    notion = NotionClient("ntn", transport=Recorder(_notion).transport)
    jira = JiraClient("https://acme.atlassian.test", "e", "t", transport=Recorder(_jira).transport)
    gmail = GmailClient(
        GmailAuth("c", "s", "r", transport=_token()), transport=Recorder(_gmail).transport
    )
    async with AsyncExitStack() as stack:
        yield {
            "notion": await stack.enter_async_context(Client(notion_mcp.create_server(notion))),
            "jira": await stack.enter_async_context(Client(jira_mcp.create_server(jira))),
            "gmail": await stack.enter_async_context(Client(gmail_mcp.create_server(gmail))),
        }


async def test_live_servers_expose_only_the_fixture_read_tools(
    live_clients: dict[Source, Client],
) -> None:
    for source, client in live_clients.items():
        tools = (await client.list_tools()).tools
        assert {(source, t.name) for t in tools} == {p for p in READ_TOOLS if p[0] == source}
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)


async def test_ingest_through_live_servers(live_clients: dict[Source, Client]) -> None:
    store = ChunkStore(QdrantClient(location=":memory:"), "live", HashEmbedder())
    reports = {r.source: r for r in await ingest(live_clients, store, max_records=1)}
    assert all(r.error is None for r in reports.values()), reports
    assert reports["notion"].truncated  # 2 pages, capped at 1: nothing may be deleted
    keys = {c.doc_key for c in store.all_chunks()}
    assert {f"notion:{PAGE_V2}", "jira:PROJ-142", f"gmail:{THREAD_ID}"} <= keys


async def test_agent_answers_from_live_records(live_clients: dict[Source, Client]) -> None:
    store = ChunkStore(QdrantClient(location=":memory:"), "live-ask", HashEmbedder())
    store.ensure_collection()
    deps = Deps(
        store=store,
        llm=MockLLM(),
        tracer=get_tracer(build_tracer_provider(Settings(), exporter=InMemorySpanExporter())),
        tools=MCPToolExecutor(live_clients, timeout_s=5),
    )
    result = await ask_async(
        "Who owns the auth migration plan?", deps, ["notion"], datetime(2026, 11, 2, tzinfo=UTC)
    )
    assert f"notion:{PAGE_V2}" in result.evidence_keys
    assert f"notion:{PAGE_V1}" in result.stale_keys  # same freshness rules on live data
    assert result.citations
    assert all((c.source, c.tool) in READ_TOOLS for c in result.tool_calls)


async def test_smoke_reports_counts_only(live_clients: dict[Source, Client]) -> None:
    for source, client in live_clients.items():
        report = await smoke(source, client)
        assert report.ok, report
        assert report.read_only
        assert report.search_results == 1
        assert report.fetched_id
        assert (report.text_chars or 0) > 0


async def test_smoke_refuses_a_server_with_write_tools() -> None:
    from servers.fixtures import jira_mcp as fixture_jira  # has update_issue (write)

    async with Client(fixture_jira.create_server()) as client:
        report = await smoke("jira", client)
    assert not report.ok
    assert "non-read-only" in (report.error or "")


async def test_gmail_consent_loopback_flow() -> None:
    def fake_browser(url: str) -> None:
        params = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        assert params["scope"] == READONLY_SCOPE
        # simulate Google redirecting the browser back to the loopback listener
        httpx.get(f"{params['redirect_uri']}/?code=abc&state={params['state']}", timeout=5)

    token = await obtain_refresh_token(
        "cid",
        "secret",
        lambda u: threading.Thread(target=fake_browser, args=(u,)).start(),
        timeout_s=10,
        transport=_token(),
    )
    assert token == "rt-new"

    with pytest.raises(LiveAPIError, match="beyond"):
        await obtain_refresh_token(
            "cid",
            "secret",
            lambda u: threading.Thread(target=fake_browser, args=(u,)).start(),
            timeout_s=10,
            transport=_token(f"{READONLY_SCOPE} https://mail.google.com/"),
        )
