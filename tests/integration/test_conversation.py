"""Short-term memory: multi-turn threads over the fixture MCP servers, checkpointed to SQLite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from sentinel.config import Settings
from sentinel.graph.build import AskResult, Deps, ask_async, thread_turns
from sentinel.graph.checkpoint import open_checkpointer
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.llm import MockLLM
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import build_tracer_provider, get_tracer
from servers.fixtures import failure_modes, gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.generate import NOW

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

BLOCKED_Q = "Which tickets are blocked by the auth migration?"
THREE_HOP = "Which tickets are blocked by the auth migration and who emailed about them?"
BLOCKED = ["jira:PROJ-142", "jira:PROJ-147", "jira:PROJ-151"]


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
def db(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints.sqlite"


def _deps(clients: dict[Source, Client], saver: Any, **overrides: object) -> Deps:
    store = ChunkStore(QdrantClient(location=":memory:"), "conv", HashEmbedder())
    store.ensure_collection()
    deps = Deps(
        store=store,
        llm=MockLLM(),
        tracer=get_tracer(build_tracer_provider(Settings(), exporter=InMemorySpanExporter())),
        tools=MCPToolExecutor(clients, timeout_s=5),
        checkpointer=saver,
    )
    for key, value in overrides.items():
        setattr(deps, key, value)
    return deps


def _sources_called(result: AskResult) -> set[str]:
    return {f"{c.source}.{c.tool}" for c in result.tool_calls}


# --- new vs resumed thread ----------------------------------------------------------------


async def test_new_thread_then_resumed_thread(clients: dict[Source, Client], db: Path) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        first = await ask_async(BLOCKED_Q, deps, as_of=NOW)
        assert first.thread_id is not None
        assert first.turn == 1
        second = await ask_async(
            "Who emailed about them?", deps, as_of=NOW, thread_id=first.thread_id
        )
        assert (second.thread_id, second.turn) == (first.thread_id, 2)
        fresh = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="other")
        assert fresh.turn == 1
        assert not any(sq.prior_keys for sq in fresh.plan)  # nothing to refer back to
        turns = await thread_turns(deps, first.thread_id)
        assert [t.question for t in turns] == [BLOCKED_Q, "Who emailed about them?"]


async def test_without_checkpointer_nothing_is_remembered(
    clients: dict[Source, Client],
) -> None:
    deps = _deps(clients, None)
    first = await ask_async(BLOCKED_Q, deps, as_of=NOW)
    follow = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="ignored")
    assert (first.thread_id, follow.thread_id, follow.turn) == (None, None, 1)
    assert not any(sq.prior_keys for sq in follow.plan)


# --- follow-ups reuse plan, evidence and unresolved items ---------------------------------


async def test_follow_up_builds_on_previous_evidence(
    clients: dict[Source, Client], db: Path
) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        first = await ask_async(BLOCKED_Q, deps, as_of=NOW, thread_id="t1")
        assert set(first.evidence_keys) >= set(BLOCKED)
        assert not [c for c in first.tool_calls if c.source == "gmail"]

        follow = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="t1")
        [sq] = follow.plan
        assert (sq.relation, sq.prior_keys) == ("mentions", BLOCKED)
        # no repeated discovery: only the email lookups for the remembered tickets
        assert _sources_called(follow) <= {"gmail.search_threads", "gmail.get_thread"}
        assert follow.hops == 1
        emails = [k for k in follow.evidence_keys if k.startswith("gmail:")]
        assert {"gmail:thr-0003", "gmail:thr-0007", "gmail:thr-0009"} <= set(emails)
        assert follow.citation_report.unknown == []


async def test_ordinal_follow_up_answers_from_memory_without_tools(
    clients: dict[Source, Client], db: Path
) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        await ask_async(BLOCKED_Q, deps, as_of=NOW, thread_id="t1")
        recall = await ask_async(
            "Who is assigned the second ticket?", deps, as_of=NOW, thread_id="t1"
        )
    [sq] = recall.plan
    assert (sq.relation, sq.prior_keys) == ("recall", ["jira:PROJ-147"])
    assert recall.tool_calls == []
    assert recall.evidence_keys == ["jira:PROJ-147"]
    assert [c.key for c in recall.citations] in ([], ["jira:PROJ-147"])


async def test_continue_resumes_unresolved_sub_questions(
    clients: dict[Source, Client], db: Path
) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver, max_hops=2)
        cut = await ask_async(THREE_HOP, deps, as_of=NOW, thread_id="t1")
        assert cut.unanswered == ["sq3"]

        resumed = await ask_async("continue", deps, as_of=NOW, thread_id="t1")
    [sq] = resumed.plan
    assert sq.relation == "mentions"
    assert sq.prior_keys == BLOCKED  # dependency taken from memory, not re-discovered
    assert _sources_called(resumed) <= {"gmail.search_threads", "gmail.get_thread"}
    assert resumed.unanswered == []
    assert {k for k in resumed.evidence_keys if k.startswith("gmail:")}


async def test_unrelated_question_in_a_thread_starts_fresh(
    clients: dict[Source, Client], db: Path
) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        await ask_async(BLOCKED_Q, deps, as_of=NOW, thread_id="t1")
        other = await ask_async(
            "What is the Pro plan price per seat?", deps, as_of=NOW, thread_id="t1"
        )
    assert other.turn == 2
    assert all(sq.relation == "search" and not sq.prior_keys for sq in other.plan)


# --- isolation and restart ----------------------------------------------------------------


async def test_threads_are_isolated(clients: dict[Source, Client], db: Path) -> None:
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        a1 = await ask_async(BLOCKED_Q, deps, as_of=NOW, thread_id="A")
        b1 = await ask_async("What is the Pro plan price per seat?", deps, as_of=NOW, thread_id="B")
        b2 = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="B")
        a2 = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="A")
        assert [len(await thread_turns(deps, t)) for t in ("A", "B")] == [2, 2]

    b_prior = {k for sq in b2.plan for k in sq.prior_keys}
    assert b_prior <= set(b1.evidence_keys)  # B only refers to B's own records
    assert not b_prior & set(a1.evidence_keys)
    assert not any(c.args.get("query") in {"PROJ-142", "PROJ-147"} for c in b2.tool_calls)
    assert a2.plan[0].prior_keys == BLOCKED  # A is unaffected by B's turns


async def test_resume_after_restart_from_the_checkpoint_file(
    clients: dict[Source, Client], db: Path
) -> None:
    async with open_checkpointer(db) as saver:
        first = await ask_async(BLOCKED_Q, _deps(clients, saver), as_of=NOW, thread_id="t1")
    assert db.exists()

    # new connection, new graph, new deps: as if the process had restarted
    async with open_checkpointer(db) as saver:
        deps = _deps(clients, saver)
        turns = await thread_turns(deps, "t1")
        assert [t.question for t in turns] == [BLOCKED_Q]
        assert turns[0].evidence_keys == first.evidence_keys
        follow = await ask_async("Who emailed about them?", deps, as_of=NOW, thread_id="t1")
        assert follow.turn == 2
        assert follow.plan[0].prior_keys == BLOCKED
        assert (await ask_async(BLOCKED_Q, deps, as_of=NOW, thread_id="new")).turn == 1
