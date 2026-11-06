"""End to end through the CLI: stdio fixture MCP subprocesses -> embedded on-disk Qdrant."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel.cli import app

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
