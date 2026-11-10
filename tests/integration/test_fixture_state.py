"""Fixture write persistence: approved mutations survive into the next server process, in a
local state directory, while the generated fixture data stays untouched."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from mcp import Client

from sentinel.mcp_clients import pool
from sentinel.mcp_clients.tools import call_json
from sentinel.permissions.actions import Action, Grant
from servers.fixtures import gmail_mcp, jira_mcp, state
from servers.fixtures.common import DEFAULT_DATA_DIR

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def data_hashes() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DEFAULT_DATA_DIR.glob("*.json"))
    }


def _grant(source: str, tool: str, args: dict[str, object]) -> Grant:
    """Tests drive the fixture server directly; the approval flow is tested elsewhere."""
    return Grant(
        approval_id="apr-test",
        action_hash=Action(source=source, tool=tool, args=args).fingerprint(),
    )


def test_the_cli_and_the_servers_agree_on_the_env_name() -> None:
    assert pool.STATE_DIR_ENV == state.STATE_DIR_ENV


async def test_jira_writes_persist_into_the_next_process(tmp_path: Path) -> None:
    before = data_hashes()
    args: dict[str, object] = {"key": "PROJ-142", "status": "Done"}
    async with Client(jira_mcp.create_server(state_dir=tmp_path)) as first:
        await call_json(first, "update_issue", args, grant=_grant("jira", "update_issue", args))
    async with Client(jira_mcp.create_server(state_dir=tmp_path)) as second:
        issue = await call_json(second, "get_issue", {"key": "PROJ-142"})
    async with Client(jira_mcp.create_server()) as no_state:
        untouched = await call_json(no_state, "get_issue", {"key": "PROJ-142"})

    assert issue["status"] == "Done"
    assert untouched["status"] == "Blocked"  # without a state dir nothing carries over
    assert (tmp_path / "jira.json").exists()
    assert data_hashes() == before


async def test_gmail_drafts_persist_and_ids_are_never_reused(tmp_path: Path) -> None:
    def draft(subject: str) -> dict[str, object]:
        return {"to": ["priya.sharma@acme.test"], "subject": subject, "body": "b"}

    async with Client(gmail_mcp.create_server(state_dir=tmp_path)) as client:
        one = await call_json(
            client,
            "create_draft",
            draft("one"),
            grant=_grant("gmail", "create_draft", draft("one")),
        )
        send = {"draft_id": one["draft_id"]}
        await call_json(client, "send", send, grant=_grant("gmail", "send", send))
    async with Client(gmail_mcp.create_server(state_dir=tmp_path)) as client:
        two = await call_json(
            client,
            "create_draft",
            draft("two"),
            grant=_grant("gmail", "create_draft", draft("two")),
        )
    assert (one["draft_id"], two["draft_id"]) == ("draft-0001", "draft-0002")
