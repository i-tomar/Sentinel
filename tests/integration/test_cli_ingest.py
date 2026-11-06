"""End to end through the CLI: stdio fixture MCP subprocesses -> embedded on-disk Qdrant."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.config import get_settings

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def embedded_qdrant(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("QDRANT_URL", str(isolated_env / "qdrant"))
    monkeypatch.delenv("SENTINEL_FIXTURE_FAILURES", raising=False)


def test_ingest_twice_then_search_with_citations() -> None:
    first = runner.invoke(app, ["ingest", "-s", "notion", "-s", "jira"])
    assert first.exit_code == 0, first.output
    assert "ok   notion  docs=40 added=40" in first.stdout
    assert "ok   jira    docs=80 added=80" in first.stdout

    second = runner.invoke(app, ["ingest", "-s", "notion", "-s", "jira"])
    assert second.exit_code == 0, second.output
    assert "docs=40 added=0 updated=0 unchanged=40 deleted=0 chunks_written=0" in second.stdout
    total = [line for line in first.stdout.splitlines() if line.startswith("collection ")]
    assert total == [line for line in second.stdout.splitlines() if line.startswith("collection ")]

    found = runner.invoke(app, ["search", "PROJ-142", "-k", "3"])
    assert found.exit_code == 0, found.output
    lines = found.stdout.splitlines()
    assert "[1] jira:PROJ-142#0" in lines[0]
    assert any(
        line.startswith("[1] Jira: PROJ-142: Migrate billing portal login to OIDC (PROJ-142) ")
        and "https://jira.fixture.acme.test/browse/PROJ-142" in line
        for line in lines
    )


def test_ask_prints_cited_answer_sources_and_trace_id() -> None:
    assert runner.invoke(app, ["ingest", "-s", "notion"]).exit_code == 0
    result = runner.invoke(app, ["ask", "Who owns the Q3 auth migration plan v2?"])
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Sources:" in out
    assert "Notion: Q3 Auth Migration Plan (v2) (auth-migration-v2)" in out
    assert re.search(r"trace_id=[0-9a-f]{32} ", out)
    assert "tracing=none llm=mock" in out

    as_json = runner.invoke(app, ["ask", "Who owns the Q3 auth migration plan v2?", "--json"])
    payload = json.loads(as_json.stdout)
    assert payload["citations"]
    assert payload["citation_report"]["unknown"] == []
    assert re.fullmatch(r"[0-9a-f]{32}", payload["trace_id"])


def test_ask_three_hop_with_live_fixture_tools() -> None:
    """No ingest: evidence comes from stdio fixture MCP servers over three hops."""
    result = runner.invoke(
        app,
        [
            "ask",
            "Which tickets are blocked by the auth migration and who emailed about them?",
            "--as-of",
            "2026-11-02",
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "hops=3 " in out
    assert "sq2 [blocked_by -> jira]" in out
    assert "sq3 [mentions -> gmail]" in out
    assert "(unanswered)" not in out
    assert "Sources:" in out
    assert re.search(r"trace_id=[0-9a-f]{32} ", out)


CHAT_FIRST = "Which tickets are blocked by the auth migration?"


def test_chat_resumes_a_thread_across_invocations() -> None:
    first = runner.invoke(app, ["chat", CHAT_FIRST, "--thread", "demo", "--as-of", "2026-11-02"])
    assert first.exit_code == 0, first.output
    assert "thread_id=demo turn=1" in first.stdout

    # a separate invocation (fresh process state) continues the same thread from the checkpoint
    second = runner.invoke(app, ["chat", "Who emailed about them?", "-t", "demo"])
    assert second.exit_code == 0, second.output
    assert "thread_id=demo turn=2" in second.stdout
    assert "sq1 [mentions -> gmail] Which emails mention PROJ-142, PROJ-147, PROJ-151?" in (
        second.stdout
    )

    history = runner.invoke(app, ["chat", "--history", "-t", "demo"])
    assert history.exit_code == 0, history.output
    assert f"[1] Q: {CHAT_FIRST}" in history.stdout
    assert "[2] Q: Who emailed about them?" in history.stdout


def test_chat_interactive_session_keeps_one_thread() -> None:
    session = runner.invoke(
        app,
        ["chat", "--as-of", "2026-11-02"],
        input=f"{CHAT_FIRST}\n\nWho is assigned the second ticket?\nexit\n",
    )
    assert session.exit_code == 0, session.output
    thread_ids = set(re.findall(r"thread_id=([0-9a-f]{32})", session.stdout))
    assert len(thread_ids) == 1
    assert "turn=1" in session.stdout
    assert "turn=2" in session.stdout
    assert "[recall -> jira]" in session.stdout


def test_chat_validates_arguments() -> None:
    assert runner.invoke(app, ["chat", "q", "--thread", "bad id!"]).exit_code == 2
    assert runner.invoke(app, ["chat", "--history"]).exit_code == 2


OWNER_Q = "Who owns the Q3 auth migration plan v2?"


def test_ask_reports_neutralized_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "Which tickets are blocked by the customer escalation process?"
    removed = runner.invoke(app, ["ask", question, "--as-of", "2026-11-02"])
    assert removed.exit_code == 0, removed.output
    assert "guard: removed 1 suspected instruction(s) from notion:customer-escalation-process" in (
        removed.stdout
    )
    assert "pre-authorized" not in removed.stdout

    monkeypatch.setenv("RETRIEVED_GUARD_MODE", "quote")
    get_settings.cache_clear()
    quoted = runner.invoke(app, ["ask", question, "--as-of", "2026-11-02"])
    assert quoted.exit_code == 0, quoted.output
    assert "guard: quoted 1 suspected instruction(s) from notion:customer-escalation-process" in (
        quoted.stdout
    )


def test_ask_memory_writes_need_the_explicit_flag() -> None:
    pending = runner.invoke(app, ["ask", OWNER_Q])
    assert pending.exit_code == 0, pending.output
    assert "memory put owns-q3-auth-migration-plan-v2: pending_approval" in pending.stdout
    assert "no memory items" in runner.invoke(app, ["memory", "list"]).stdout

    written = runner.invoke(app, ["ask", OWNER_Q, "--auto-approve-memory"])
    assert written.exit_code == 0, written.output
    assert "memory put owns-q3-auth-migration-plan-v2: written" in written.stdout
    listed = runner.invoke(app, ["memory", "list"]).stdout
    assert "owns-q3-auth-migration-plan-v2:" in listed
    assert "verified" in listed

    recalled = runner.invoke(app, ["ask", "Who owns the auth migration plan?"])
    assert "memory note [verified] owns-q3-auth-migration-plan-v2" in recalled.stdout


def test_auto_approve_is_refused_on_live_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_MODE", "live")
    result = runner.invoke(app, ["ask", OWNER_Q, "--no-tools", "--auto-approve-memory"])
    assert result.exit_code == 2
    assert "refused with DATA_MODE=live" in result.output


def test_memory_cli_put_search_mark_stale() -> None:
    put = runner.invoke(app, ["memory", "put", "cutover-date", "Cutover is on 2026-11-20"])
    assert put.exit_code == 0, put.output
    assert "source=user" in put.stdout
    assert "unverified" in put.stdout
    assert "cutover-date" in runner.invoke(app, ["memory", "search", "when is cutover"]).stdout
    stale = runner.invoke(app, ["memory", "mark-stale", "cutover-date", "--reason", "moved"])
    assert "stale_reason=moved" in stale.stdout
    assert "no memory items" in runner.invoke(app, ["memory", "search", "cutover"]).stdout
    assert runner.invoke(app, ["memory", "mark-stale", "nope"]).exit_code == 1
    assert runner.invoke(app, ["memory", "put", "Bad Key", "x"]).exit_code == 2


def test_ask_without_evidence_says_so() -> None:
    assert runner.invoke(app, ["ingest", "-s", "notion"]).exit_code == 0
    result = runner.invoke(app, ["ask", "What is the capital of France?"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("I don't have enough evidence")
    assert "Sources:" not in result.stdout
    assert "trace_id=" in result.stdout


def test_ingest_reports_injected_source_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_FIXTURE_FAILURES", "gmail=rate_limit")
    result = runner.invoke(app, ["ingest", "-s", "gmail"])
    assert result.exit_code == 1
    assert "FAIL gmail" in result.stdout
    assert "429" in result.stdout
