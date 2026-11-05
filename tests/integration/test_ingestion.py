"""Ingest the fixture MCP servers into embedded Qdrant (HashEmbedder: offline, deterministic)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta

import pytest
from mcp import Client
from qdrant_client import QdrantClient

from sentinel.citations import CitationResolver
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.ingest import ingest
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from servers.fixtures import failure_modes, gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.common import DEFAULT_DATA_DIR, load_gmail, load_jira, load_notion

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

T1 = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
T2 = T1 + timedelta(hours=6)
NOTION, JIRA, GMAIL = load_notion(), load_jira(), load_gmail()
EXPECTED_KEYS = (
    {f"notion:{p.id}" for p in NOTION}
    | {f"jira:{i.key}" for i in JIRA}
    | {f"gmail:{t.id}" for t in GMAIL}
)


@pytest.fixture(autouse=True)
def no_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(failure_modes.FAILURES_ENV, raising=False)


@pytest.fixture
async def clients() -> AsyncIterator[dict[Source, Client]]:
    async with AsyncExitStack() as stack:
        yield {
            "notion": await stack.enter_async_context(Client(notion_mcp.create_server())),
            "jira": await stack.enter_async_context(Client(jira_mcp.create_server())),
            "gmail": await stack.enter_async_context(Client(gmail_mcp.create_server())),
        }


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(QdrantClient(location=":memory:"), "it", HashEmbedder())


@pytest.fixture
async def indexed(clients: dict[Source, Client], store: ChunkStore) -> ChunkStore:
    reports = await ingest(clients, store, fetched_at=T1)
    assert all(r.error is None for r in reports), reports
    return store


def _keys(store: ChunkStore) -> set[str]:
    return {c.doc_key for c in store.all_chunks()}


# --- ingestion ----------------------------------------------------------------------------


async def test_first_ingest_indexes_every_record_once(
    clients: dict[Source, Client], store: ChunkStore
) -> None:
    reports = {r.source: r for r in await ingest(clients, store, fetched_at=T1)}
    assert {s: (r.documents, r.added) for s, r in reports.items()} == {
        "notion": (40, 40),
        "jira": (80, 80),
        "gmail": (60, 60),
    }
    chunks = store.all_chunks()
    assert {c.doc_key for c in chunks} == EXPECTED_KEYS
    assert len(chunks) == store.count() == sum(r.chunks_written for r in reports.values())
    assert len({c.chunk_id for c in chunks}) == len(chunks)


async def test_reingest_is_idempotent(clients: dict[Source, Client], indexed: ChunkStore) -> None:
    before = {c.chunk_id: c for c in indexed.all_chunks()}
    reports = await ingest(clients, indexed, fetched_at=T2)
    assert all((r.added, r.updated, r.deleted, r.chunks_written) == (0, 0, 0, 0) for r in reports)
    assert sum(r.unchanged for r in reports) == 180
    after = {c.chunk_id: c for c in indexed.all_chunks()}
    assert after.keys() == before.keys()
    assert all(c.fetched_at == T2 for c in after.values())  # re-fetch recorded
    assert all(after[k].text == before[k].text for k in before)


async def test_changed_record_is_replaced_not_duplicated(
    clients: dict[Source, Client], indexed: ChunkStore
) -> None:
    count = indexed.count()
    result = await clients["jira"].call_tool(
        "update_issue", {"key": "PROJ-142", "status": "In Progress", "comment": "tenant is live"}
    )
    assert not result.is_error
    [jira] = [r for r in await ingest({"jira": clients["jira"]}, indexed, fetched_at=T2)]
    assert (jira.updated, jira.added, jira.unchanged) == (1, 0, 79)
    chunks = [c for c in indexed.all_chunks() if c.doc_key == "jira:PROJ-142"]
    assert any("Status: In Progress" in c.text for c in chunks)
    assert not any("Status: Blocked" in c.text for c in chunks)
    assert indexed.count() in range(count, count + 2)  # same doc, maybe one more chunk
    hits = indexed.search("tenant is live", 3, sources=["jira"], mode="keyword")
    assert hits[0].chunk.doc_key == "jira:PROJ-142"


async def test_record_removed_at_source_is_deleted(indexed: ChunkStore) -> None:
    fewer = [p for p in NOTION if p.id != "hiring-plan-h2"]
    async with Client(notion_mcp.create_server(fewer)) as notion:
        [report] = await ingest({"notion": notion}, indexed, fetched_at=T2)
    assert (report.deleted, report.unchanged) == (1, 39)
    assert "notion:hiring-plan-h2" not in _keys(indexed)


async def test_failing_source_leaves_index_untouched(
    monkeypatch: pytest.MonkeyPatch, clients: dict[Source, Client], indexed: ChunkStore
) -> None:
    gmail_before = indexed.count("gmail")
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "gmail=error_500,notion.get_page=malformed")
    reports = {r.source: r for r in await ingest(clients, indexed, fetched_at=T2)}
    assert reports["gmail"].error is not None
    assert "500" in reports["gmail"].error
    assert reports["notion"].error is not None
    assert "malformed" in reports["notion"].error
    assert reports["jira"].error is None
    assert indexed.count("gmail") == gmail_before
    assert {k for k in _keys(indexed) if k.startswith("notion:")} == {
        f"notion:{p.id}" for p in NOTION
    }


async def test_ground_truth_is_not_indexed(indexed: ChunkStore) -> None:
    manifest = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest_only = [p["attacker_goal"] for p in manifest["poisoned"]] + [
        b["why_it_looks_suspicious"] for b in manifest["benign_lookalikes"]
    ]
    raw_records = "".join(
        (DEFAULT_DATA_DIR / f).read_text(encoding="utf-8")
        for f in ("notion_pages.json", "jira_issues.json", "gmail_threads.json")
    )
    corpus = "\n".join(c.text for c in indexed.all_chunks())
    for text in manifest_only:
        assert text not in raw_records  # precondition: string exists only in the manifest
        assert text not in corpus
    assert _keys(indexed) == EXPECTED_KEYS
    assert not any(k.split(":", 1)[1].startswith("manifest") for k in _keys(indexed))


# --- timestamps ---------------------------------------------------------------------------


async def test_source_timestamps_are_preserved(indexed: ChunkStore) -> None:
    expected = (
        {f"notion:{p.id}": p.updated_at for p in NOTION}
        | {f"jira:{i.key}": i.updated_at for i in JIRA}
        | {f"gmail:{t.id}": t.messages[-1].sent_at for t in GMAIL}
    )
    for chunk in indexed.all_chunks():
        assert chunk.updated_at == expected[chunk.doc_key], chunk.doc_key
        assert chunk.fetched_at == T1


async def test_stale_and_current_docs_both_retrievable_with_dates(indexed: ChunkStore) -> None:
    hits = indexed.search("Pro plan price per seat per month", 4, sources=["notion"])
    by_key = {h.chunk.doc_key: h.chunk for h in hits}
    assert {"notion:pricing-2025", "notion:pricing-2026"} <= by_key.keys()
    assert by_key["notion:pricing-2026"].updated_at > by_key["notion:pricing-2025"].updated_at


# --- ranking ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "keyword"])
async def test_exact_ticket_key_ranks_its_ticket_first(indexed: ChunkStore, mode: str) -> None:
    hits = indexed.search("PROJ-142", 5, mode=mode)  # type: ignore[arg-type]
    assert hits[0].chunk.doc_key == "jira:PROJ-142"


async def test_hybrid_ranking_on_story_questions(indexed: ChunkStore) -> None:
    top3 = [h.chunk.doc_key for h in indexed.search("Q3 auth migration plan cutover owner", 3)]
    assert "notion:auth-migration-v2" in top3
    gmail = indexed.search("SSO login failing field team", 3, sources=["gmail"])
    assert gmail[0].chunk.doc_key == "gmail:thr-0003"
    assert {h.chunk.source for h in gmail} == {"gmail"}
    oncall = [h.chunk.doc_key for h in indexed.search("SEV1 acknowledged within minutes", 2)]
    assert set(oncall) == {"notion:oncall-runbook-2025", "notion:oncall-runbook-2026"}


async def test_hybrid_scores_are_descending(indexed: ChunkStore) -> None:
    hits = indexed.search("invoice", 8)
    assert len(hits) == 8
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


# --- citations ----------------------------------------------------------------------------


async def test_citations_resolve_to_returned_records(indexed: ChunkStore) -> None:
    records = (
        {f"notion:{p.id}": (p.title, p.url) for p in NOTION}
        | {f"jira:{i.key}": (f"{i.key}: {i.summary}", i.url) for i in JIRA}
        | {f"gmail:{t.id}": (t.subject, t.url) for t in GMAIL}
    )
    resolver = CitationResolver()
    for hit in indexed.search("tickets blocked by PROJ-120 auth migration", 6):
        resolver.add(hit.chunk)
    assert resolver.citations
    for c in resolver.citations:
        assert (c.title, c.url) == records[c.key]
        assert c.fetched_at == T1
        assert c.snippet
    answer = "Blocked work [1][2]; invented source [42]."
    check = resolver.check(answer)
    assert check.unknown == [42]
    assert resolver.resolve(42) is None
