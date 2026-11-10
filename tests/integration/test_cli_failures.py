"""Tool failures through the CLI with real stdio fixture servers: no crash, no traceback,
an honest answer, and uncertain writes reported and audited as uncertain."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel.cli import app
from servers.fixtures import failure_modes

pytestmark = pytest.mark.integration

runner = CliRunner()
THREE_HOP = "Which tickets are blocked by the auth migration and who emailed about them?"


@pytest.fixture(autouse=True)
def embedded_qdrant(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("QDRANT_URL", str(isolated_env / "qdrant"))
    monkeypatch.setenv("TOOL_TIMEOUT_S", "5")
    monkeypatch.delenv(failure_modes.FAILURES_ENV, raising=False)


def _ask(question: str) -> str:
    result = runner.invoke(app, ["ask", question, "--as-of", "2026-11-02"])
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    return result.output


def test_a_server_that_fails_to_start_does_not_stop_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "gmail.startup=crash")
    out = _ask(THREE_HOP)
    assert "I could not read gmail" in out
    assert "jira:PROJ-142" in out or "PROJ-142" in out


def test_every_source_failing_gives_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "*=error_500")
    out = _ask(THREE_HOP)
    assert "none of the sources could be read" in out
    assert "Sources:" not in out


def test_ingest_reports_a_server_that_fails_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "gmail.startup=crash")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "FAIL gmail" in result.stdout
    assert "ok   notion" in result.stdout
    assert "Traceback" not in result.output


def test_a_write_interrupted_by_a_crash_is_reported_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _ask("Set PROJ-142 to Done.")
    approval = re.search(r"id=(apr-[0-9a-f]{16})", out)
    session = re.search(r"--session ([0-9a-f]{32})", out)
    assert approval
    assert session
    monkeypatch.setenv(failure_modes.FAILURES_ENV, "jira.update_issue=crash")
    result = runner.invoke(app, ["approve", approval.group(1), "--session", session.group(1)])
    assert result.exit_code == 1
    assert "uncertain" in result.output
    assert "may or may not have" in result.output
    assert "Traceback" not in result.output

    again = runner.invoke(app, ["approve", approval.group(1), "--session", session.group(1)])
    assert again.exit_code == 1
    assert "already used" in again.output  # never retried automatically
    records = [json.loads(line) for line in Path(".data/audit.jsonl").read_text().splitlines()]
    events = [r["event"] for r in records if r["approval_id"] == approval.group(1)]
    assert events[-3:] == ["attempted", "uncertain", "blocked"]
