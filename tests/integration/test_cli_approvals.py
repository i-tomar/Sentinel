"""Approval flow through the CLI: `ask` proposes, `approve`/`deny` decide, stdio fixture servers.

Writes land only in a short-lived fixture server process; nothing reaches a live account.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anyio
import pytest
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.config import get_settings
from sentinel.mcp_clients.pool import open_clients
from sentinel.mcp_clients.tools import call_json
from tests.integration.test_fixture_state import data_hashes

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def embedded_qdrant(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("QDRANT_URL", str(isolated_env / "qdrant"))
    monkeypatch.delenv("SENTINEL_FIXTURE_FAILURES", raising=False)


def _proposal(question: str) -> tuple[str, str, str]:
    result = runner.invoke(app, ["ask", question, "--as-of", "2026-11-02"])
    assert result.exit_code == 0, result.output
    approval = re.search(r"id=(apr-[0-9a-f]{16})", result.stdout)
    session = re.search(r"--session ([0-9a-f]{32})", result.stdout)
    assert approval
    assert session
    return result.stdout, approval.group(1), session.group(1)


def _audit() -> list[dict[str, object]]:
    lines = Path(".data/audit.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_ask_proposes_and_approve_runs_exactly_once() -> None:
    out, approval_id, session = _proposal("Set PROJ-142 to Done.")
    assert "approval required (write): jira.update_issue key=PROJ-142 status=Done" in out
    assert "pending" in runner.invoke(app, ["approvals", "--session", session]).stdout

    elsewhere = runner.invoke(app, ["approve", approval_id, "--session", "another-session"])
    assert elsewhere.exit_code == 1
    assert "another session" in elsewhere.output

    done = runner.invoke(app, ["approve", approval_id, "--session", session])
    assert done.exit_code == 0, done.output
    assert "done: jira.update_issue key=PROJ-142 status=Done" in done.stdout

    replay = runner.invoke(app, ["approve", approval_id, "--session", session])
    assert replay.exit_code == 1
    assert "already used" in replay.output

    events = [r["event"] for r in _audit() if r["approval_id"] == approval_id]
    assert events == ["proposed", "blocked", "approved", "attempted", "succeeded", "blocked"]
    raw = Path(".data/audit.jsonl").read_text(encoding="utf-8")
    assert "Set PROJ-142 to Done" not in raw  # the question itself is never logged


async def _read_issue(key: str) -> dict[str, object]:
    """Read through a fresh fixture server process, launched exactly as the CLI launches it."""
    async with open_clients(get_settings(), ["jira"]) as clients:
        issue: dict[str, object] = await call_json(clients["jira"], "get_issue", {"key": key})
        return issue


def test_approved_write_is_visible_to_later_commands() -> None:
    before = data_hashes()
    assert anyio.run(_read_issue, "PROJ-142")["status"] == "Blocked"  # fresh, isolated state

    _, approval_id, session = _proposal("Set PROJ-142 to Done.")
    assert anyio.run(_read_issue, "PROJ-142")["status"] == "Blocked"  # proposing changed nothing
    done = runner.invoke(app, ["approve", approval_id, "--session", session])
    assert done.exit_code == 0, done.output

    assert anyio.run(_read_issue, "PROJ-142")["status"] == "Done"
    later = runner.invoke(
        app, ["ask", "What is the status of PROJ-142?", "--as-of", "2026-11-02", "--json"]
    )
    assert later.exit_code == 0, later.output
    payload = json.loads(later.stdout)
    assert "jira:PROJ-142" in payload["evidence_keys"]
    [cited] = [c for c in payload["citations"] if c["doc_id"] == "PROJ-142"]
    assert "Status: Done." in cited["snippet"]  # the agent read the changed record
    assert Path(".data/fixture-state/jira.json").exists()  # inside this test's tmp dir
    assert data_hashes() == before  # generated fixture data untouched


def test_memory_write_waits_for_approval_through_the_cli() -> None:
    out, _, session = _proposal("Set PROJ-142 to Done.")
    memory = re.search(r"memory put \S+: pending_approval .* approval=(apr-[0-9a-f]{16})", out)
    assert memory, out
    assert "no memory items" in runner.invoke(app, ["memory", "list"]).stdout
    done = runner.invoke(app, ["approve", memory.group(1), "--session", session])
    assert done.exit_code == 0, done.output
    assert "done: memory.put_memory" in done.stdout
    assert "no memory items" not in runner.invoke(app, ["memory", "list"]).stdout


def test_deny_and_typed_confirmation_for_destructive_actions() -> None:
    out, approval_id, session = _proposal("Send draft-0001.")
    assert "approval required (destructive): gmail.send draft_id=draft-0001" in out
    assert '--confirm "send draft-0001"' in out

    unconfirmed = runner.invoke(app, ["approve", approval_id, "--session", session])
    assert unconfirmed.exit_code == 1
    assert "typed confirmation" in unconfirmed.output

    denied = runner.invoke(app, ["deny", approval_id, "--session", session])
    assert denied.exit_code == 0, denied.output
    later = runner.invoke(
        app, ["approve", approval_id, "--session", session, "--confirm", "send draft-0001"]
    )
    assert later.exit_code == 1
    assert "denied" in later.output


def test_policy_refuses_out_of_domain_recipients_without_asking() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "Draft an email to partner@globex.test about the cutover date.",
            "--as-of",
            "2026-11-02",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "action refused by policy: gmail.create_draft" in result.stdout
    assert "outside ALLOWED_EMAIL_DOMAINS" in result.stdout
    assert "gmail.create_draft" not in runner.invoke(app, ["approvals"]).stdout
