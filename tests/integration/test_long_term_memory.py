"""Long-term memory end to end: fixture MCP servers, SQLite memory, MockLLM, no credentials.

The fixture clock is 2026-11-02. auth-migration-v1 (owner Vikram Rao) is superseded by v2
(owner Priya Sharma); PROJ-142 is Blocked and was last updated 2026-10-31.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from sentinel.config import Settings
from sentinel.graph.build import Deps, ask_async
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.llm import MockLLM
from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import build_tracer_provider, get_tracer
from servers.fixtures import failure_modes, gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.common import load_jira, load_notion
from servers.fixtures.generate import NOW

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

OWNER_Q = "Who owns the Q3 auth migration plan v2?"
PAGES = {p.id: p for p in load_notion()}
ISSUES = {i.key: i for i in load_jira()}


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
def memory(tmp_path: Path) -> SQLiteMemoryBackend:
    return SQLiteMemoryBackend(tmp_path / "memory.sqlite")


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _deps(
    clients: dict[Source, Client],
    memory: SQLiteMemoryBackend,
    exporter: InMemorySpanExporter | None = None,
    **overrides: Any,
) -> Deps:
    store = ChunkStore(QdrantClient(location=":memory:"), "ltm", HashEmbedder())
    store.ensure_collection()
    deps = Deps(
        store=store,
        llm=MockLLM(),
        tracer=get_tracer(
            build_tracer_provider(Settings(), exporter=exporter or InMemorySpanExporter())
        ),
        tools=MCPToolExecutor(clients, timeout_s=5),
        memory=memory,
    )
    for key, value in overrides.items():
        setattr(deps, key, value)
    return deps


def _seed(memory: SQLiteMemoryBackend, **fields: Any) -> MemoryItem:
    return memory.put(MemoryItem.model_validate(fields))


# --- write policy -------------------------------------------------------------------------


async def test_verified_fact_saved_with_auto_approve_and_recalled_in_a_new_session(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    first = await ask_async(OWNER_Q, _deps(clients, memory, memory_auto_approve=True), as_of=NOW)
    [write] = [w for w in first.memory_writes if w.op == "put"]
    assert write.status == "written"
    stored = memory.get(write.key)
    assert stored is not None
    assert stored.verified is True
    assert stored.source in {c.key for c in first.citations}
    assert stored.updated_at == next(
        c.updated_at for c in first.citations if c.key == stored.source
    )
    assert stored.record_keys == [c.key for c in first.citations][:8]

    # a different session (no shared thread) recalls it and re-reads its source record
    later = await ask_async("Who owns the auth migration plan?", _deps(clients, memory), as_of=NOW)
    assert stored.key in [n.key for n in later.memory_notes]
    refresh = [c for c in later.tool_calls if c.sub_question_id == "memory"]
    assert refresh
    assert {f"{c.source}:{next(iter(c.args.values()))}" for c in refresh} >= {stored.source}


async def test_writes_need_approval_without_the_flag(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    result = await ask_async(OWNER_Q, _deps(clients, memory), as_of=NOW)
    assert [w.status for w in result.memory_writes] == ["pending_approval"]
    assert memory.all() == []


async def test_live_mode_always_requires_approval(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    deps = _deps(clients, memory, data_mode="live")
    result = await ask_async(OWNER_Q, deps, as_of=NOW)
    assert [w.status for w in result.memory_writes] == ["pending_approval"]
    assert memory.all() == []
    deps.memory_auto_approve = True  # Settings refuses this combination; the policy does too
    with pytest.raises(ValueError, match="only allowed with DATA_MODE=fixture"):
        await ask_async(OWNER_Q, deps, as_of=NOW)


# --- freshness: memory vs newer source records --------------------------------------------


async def test_memory_from_a_superseded_doc_loses_to_the_newer_doc(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    v1 = PAGES["auth-migration-v1"]
    _seed(
        memory,
        key="q3-plan-owner",
        value="Vikram Rao owns the Q3 auth migration plan",
        source="notion:auth-migration-v1",
        verified=True,
        updated_at=v1.updated_at,
        confidence=0.9,
    )
    question = "Who owns the Q3 auth migration plan?"
    result = await ask_async(question, _deps(clients, memory, memory_auto_approve=True), as_of=NOW)
    [note] = [n for n in result.memory_notes if n.key == "q3-plan-owner"]
    assert note.stale is True
    assert note.superseded_by == "notion:auth-migration-v2"
    assert ("mark_stale", "written") in {(w.op, w.status) for w in result.memory_writes}
    persisted = memory.get("q3-plan-owner")
    assert persisted is not None
    assert persisted.stale is True
    assert persisted.superseded_by == "notion:auth-migration-v2"
    # the fresh answer is remembered as its own verified fact, sourced from the newer doc
    fresh = memory.get("owns-q3-auth-migration-plan")
    assert fresh is not None
    assert fresh.verified is True
    assert "notion:auth-migration-v1" not in fresh.record_keys
    assert "Vikram" not in result.answer
    assert "notion:auth-migration-v2" in result.evidence_keys
    assert "notion:auth-migration-v1" not in {c.key for c in result.citations}


async def test_newer_source_record_wins_even_while_stale_marking_awaits_approval(
    clients: dict[Source, Client],
    memory: SQLiteMemoryBackend,
    exporter: InMemorySpanExporter,
) -> None:
    issue = ISSUES["PROJ-142"]
    _seed(
        memory,
        key="status-proj-142",
        value="PROJ-142 is in progress and nearly done",
        source="jira:PROJ-142",
        verified=True,
        updated_at=issue.updated_at - timedelta(days=20),  # remembered before the last change
        confidence=0.9,
    )
    result = await ask_async(
        "What is the status of PROJ-142?", _deps(clients, memory, exporter), as_of=NOW
    )
    refresh = [c for c in result.tool_calls if c.sub_question_id == "memory"]
    assert [(c.tool, c.args) for c in refresh] == [("get_issue", {"key": "PROJ-142"})]
    [note] = [n for n in result.memory_notes if n.key == "status-proj-142"]
    assert note.stale is True
    assert "no longer supports" in (note.stale_reason or "")
    assert ("mark_stale", "pending_approval") in {(w.op, w.status) for w in result.memory_writes}
    persisted = memory.get("status-proj-142")
    assert persisted is not None
    assert persisted.stale is False  # not written without approval...
    prompt = "\n".join(
        str(s.attributes.get("langfuse.observation.input", ""))
        for s in exporter.get_finished_spans()
        if s.name == "llm.generate" and s.attributes
    )
    assert "nearly done" not in prompt  # ...but the stale note never reaches the model
    assert "jira:PROJ-142" in result.evidence_keys  # the fresh record is evidence instead


async def test_memory_that_still_matches_its_source_stays_fresh(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    issue = ISSUES["PROJ-142"]
    _seed(
        memory,
        key="status-proj-142",
        value="PROJ-142 is blocked",
        source="jira:PROJ-142",
        verified=True,
        updated_at=issue.updated_at - timedelta(days=20),
    )
    result = await ask_async("What is the status of PROJ-142?", _deps(clients, memory), as_of=NOW)
    [note] = [n for n in result.memory_notes if n.key == "status-proj-142"]
    assert note.stale is False
    assert not [w for w in result.memory_writes if w.op == "mark_stale"]


# --- unverified memory is never cited as fact ---------------------------------------------


async def test_unverified_memory_is_labelled_and_never_stated_as_fact(
    clients: dict[Source, Client],
    memory: SQLiteMemoryBackend,
    exporter: InMemorySpanExporter,
) -> None:
    _seed(
        memory,
        key="auth-migration-cutover-date",
        value="The auth migration cutover moved to 2027-03-01",
        source="user",
        verified=False,
        updated_at=NOW,
    )
    question = "When is the auth migration cutover date?"
    dry = MockLLM()
    await ask_async(question, _deps(clients, memory, exporter, llm=dry), as_of=NOW)
    script = (
        "The auth migration cutover moved to 2027-03-01 [1]. "
        "The plan lists a cutover date of 2026-11-20 [1]."
    )
    llm = MockLLM(scripted={dry.calls[-1]: script})
    result = await ask_async(question, _deps(clients, memory, exporter, llm=llm), as_of=NOW)

    assert llm.calls  # the scripted (memory-parroting) reply was used
    assert result.citation_report.unverified_memory_claims
    assert "2027-03-01" not in result.answer.split("(Removed")[0]
    assert "2026-11-20" in result.answer
    assert not [w for w in result.memory_writes if w.op == "put"]  # flagged answer not saved
    prompts = [
        str(s.attributes.get("langfuse.observation.input", ""))
        for s in exporter.get_finished_spans()
        if s.name == "llm.generate" and s.attributes
    ]
    assert any('<note verified="false" source="user"' in p for p in prompts)


async def test_no_memory_backend_means_no_recall_and_no_writes(
    clients: dict[Source, Client], memory: SQLiteMemoryBackend
) -> None:
    _seed(
        memory, key="owns-q3-auth-migration-plan-v2", value="Priya", source="user", updated_at=NOW
    )
    deps = _deps(clients, memory)
    deps.memory = None
    result = await ask_async(OWNER_Q, deps, as_of=NOW)
    assert result.memory_notes == []
    assert result.memory_writes == []
