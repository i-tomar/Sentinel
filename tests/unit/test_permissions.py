"""Permission layer units: registry, policy, approval store, audit log, MCP call choke point.

Written before the implementation (step l). Clock and storage are injected: no sleeps, no files
outside tmp_path, no network.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.types import TextContent, ToolAnnotations

from sentinel.mcp_clients.registry import READ_TOOLS, REGISTRY, lookup, spec_by_name
from sentinel.mcp_clients.tools import call_json
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.permissions.actions import Action, Grant, PermissionDenied
from sentinel.permissions.approvals import ApprovalError, ApprovalStore
from sentinel.permissions.audit import AuditLog
from sentinel.permissions.gate import ActionOutcome, ActionRunner
from sentinel.permissions.policy import Policy
from servers.fixtures import gmail_mcp, jira_mcp, notion_mcp
from servers.memory_mcp import server as memory_mcp

pytestmark = pytest.mark.unit

T0 = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
DONE = Action(source="jira", tool="update_issue", args={"key": "PROJ-142", "status": "Done"})
DRAFT = Action(
    source="gmail",
    tool="create_draft",
    args={"to": ["priya.sharma@acme.test"], "subject": "Cutover", "body": "Secret plan B."},
)
SEND = Action(source="gmail", tool="send", args={"draft_id": "draft-0001"})


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> ApprovalStore:
    return ApprovalStore(":memory:", ttl_s=600, clock=clock)


# --- registry: trusted metadata -----------------------------------------------------------


async def _annotations(server: Any) -> dict[str, ToolAnnotations | None]:
    async with Client(server) as client:
        return {t.name: t.annotations for t in (await client.list_tools()).tools}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "server"),
    [
        ("notion", notion_mcp.create_server()),
        ("jira", jira_mcp.create_server()),
        ("gmail", gmail_mcp.create_server()),
    ],
)
async def test_every_fixture_tool_is_classified_like_its_server_says(
    source: str, server: Any
) -> None:
    for name, ann in (await _annotations(server)).items():
        spec = lookup(source, name)
        assert spec is not None, f"{source}.{name} is not in the registry"
        assert ann is not None
        expected = (
            "read" if ann.read_only_hint else "destructive" if ann.destructive_hint else "write"
        )
        assert spec.risk == expected, f"{source}.{name}"


@pytest.mark.anyio
async def test_memory_server_tools_are_classified(tmp_path: Path) -> None:
    server = memory_mcp.create_server(SQLiteMemoryBackend(tmp_path / "m.sqlite"))
    for name, ann in (await _annotations(server)).items():
        spec = lookup("memory", name)
        assert spec is not None
        assert ann is not None
        assert spec.risk == ("read" if ann.read_only_hint else "write")


def test_tool_names_are_unique_and_reads_are_derived_from_risk() -> None:
    names = [name for _, name in REGISTRY]
    assert len(names) == len(set(names))
    assert {k for k, s in REGISTRY.items() if s.risk == "read"} >= READ_TOOLS
    assert ("jira", "update_issue") not in READ_TOOLS
    assert spec_by_name("send") is lookup("gmail", "send")
    assert lookup("gmail", "delete_everything") is None


# --- policy -------------------------------------------------------------------------------


def test_policy_verdicts_by_risk() -> None:
    policy = Policy()
    read = Action(source="jira", tool="get_issue", args={"key": "PROJ-1"})
    assert policy.evaluate(read, "fixture").verdict == "allow"
    assert policy.evaluate(DONE, "fixture").verdict == "require_approval"
    send = policy.evaluate(SEND, "fixture")
    assert send.verdict == "require_approval"
    assert send.needs_confirmation


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (Action(source="gmail", tool="forward_all", args={}), "unknown"),
        (Action(source="slack", tool="post", args={}), "unknown"),
        (Action(source="jira", tool="send", args={"draft_id": "draft-0001"}), "unknown"),
        (Action(source="jira", tool="update_issue", args={"key": "PROJ-142"}), "invalid"),
        (
            Action(source="jira", tool="update_issue", args={"key": "x; drop", "status": "Done"}),
            "invalid",
        ),
        (
            Action(
                source="jira",
                tool="update_issue",
                args={"key": "PROJ-1", "status": "Done", "extra": "field"},
            ),
            "invalid",
        ),
        (
            Action(
                source="gmail",
                tool="create_draft",
                args={"to": ["dana.w@evil.test"], "subject": "s", "body": "b"},
            ),
            "outside ALLOWED_EMAIL_DOMAINS",
        ),
        (
            Action(
                source="gmail",
                tool="create_draft",
                args={"to": [f"u{i}@acme.test" for i in range(11)], "subject": "s", "body": "b"},
            ),
            "invalid",
        ),
        (
            Action(
                source="jira",
                tool="update_issue",
                args={"key": "PROJ-1", "assignee": "x@evil.test"},
            ),
            "outside ALLOWED_EMAIL_DOMAINS",
        ),
    ],
)
def test_policy_denies_without_asking(action: Action, reason: str) -> None:
    decision = Policy().evaluate(action, "fixture")
    assert decision.verdict == "deny"
    assert reason in decision.reason


def test_live_source_writes_are_denied_but_memory_writes_are_not() -> None:
    assert Policy().evaluate(DONE, "live").verdict == "deny"
    note = Action(source="memory", tool="mark_stale", args={"key": "owner", "reason": "superseded"})
    assert Policy().evaluate(note, "live").verdict == "require_approval"


def test_fingerprint_covers_tool_and_every_argument() -> None:
    same = Action(source="jira", tool="update_issue", args={"status": "Done", "key": "PROJ-142"})
    assert same.fingerprint() == DONE.fingerprint()
    changed = Action(source="jira", tool="update_issue", args={"key": "PROJ-147", "status": "Done"})
    assert changed.fingerprint() != DONE.fingerprint()


# --- approval store -----------------------------------------------------------------------


def _approved(store: ApprovalStore, action: Action = DONE, session: str = "s1") -> str:
    approval = store.request(action, risk="write", session=session, principal="ishan")
    store.approve(approval.id, session=session, principal="ishan")
    return approval.id


def test_approved_action_can_be_consumed_once(store: ApprovalStore) -> None:
    approval_id = _approved(store)
    grant = store.consume(approval_id, DONE, session="s1", principal="ishan")
    assert grant.action_hash == DONE.fingerprint()
    with pytest.raises(ApprovalError, match="already used"):
        store.consume(approval_id, DONE, session="s1", principal="ishan")
    record = store.get(approval_id)
    assert record is not None
    assert record.status == "consumed"
    assert record.args is None  # arguments are not kept once the approval is closed


def test_pending_or_missing_approval_cannot_execute(store: ApprovalStore) -> None:
    pending = store.request(DONE, risk="write", session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="not approved"):
        store.consume(pending.id, DONE, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="no such approval"):
        store.consume("apr-missing", DONE, session="s1", principal="ishan")


def test_denied_approval_cannot_execute_or_be_approved_later(store: ApprovalStore) -> None:
    pending = store.request(DONE, risk="write", session="s1", principal="ishan")
    store.deny(pending.id, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="denied"):
        store.consume(pending.id, DONE, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="denied"):
        store.approve(pending.id, session="s1", principal="ishan")


def test_changed_arguments_need_a_new_approval(store: ApprovalStore) -> None:
    approval_id = _approved(store)
    other = Action(source="jira", tool="update_issue", args={"key": "PROJ-147", "status": "Done"})
    with pytest.raises(ApprovalError, match="arguments changed"):
        store.consume(approval_id, other, session="s1", principal="ishan")
    other_tool = Action(source="gmail", tool="send", args={"draft_id": "draft-0001"})
    with pytest.raises(ApprovalError, match="arguments changed"):
        store.consume(approval_id, other_tool, session="s1", principal="ishan")
    assert store.consume(approval_id, DONE, session="s1", principal="ishan")  # still usable


def test_approvals_expire(store: ApprovalStore, clock: Clock) -> None:
    pending = store.request(DONE, risk="write", session="s1", principal="ishan")
    approved_id = _approved(store, session="s1")
    clock.now = T0 + timedelta(seconds=601)
    with pytest.raises(ApprovalError, match="expired"):
        store.approve(pending.id, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="expired"):
        store.consume(approved_id, DONE, session="s1", principal="ishan")
    record = store.get(approved_id)
    assert record is not None
    assert record.status == "expired"


def test_approvals_are_bound_to_session_and_user(store: ApprovalStore) -> None:
    pending = store.request(DONE, risk="write", session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="another session"):
        store.approve(pending.id, session="s2", principal="ishan")
    with pytest.raises(ApprovalError, match="another user"):
        store.approve(pending.id, session="s1", principal="mallory")
    approval_id = _approved(store, session="s1")
    with pytest.raises(ApprovalError, match="another session"):
        store.consume(approval_id, DONE, session="s2", principal="ishan")
    with pytest.raises(ApprovalError, match="another user"):
        store.consume(approval_id, DONE, session="s1", principal="mallory")


def test_destructive_actions_need_the_typed_confirmation(store: ApprovalStore) -> None:
    pending = store.request(
        SEND, risk="destructive", session="s1", principal="ishan", confirmation="send draft-0001"
    )
    with pytest.raises(ApprovalError, match="confirmation"):
        store.approve(pending.id, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="confirmation"):
        store.approve(pending.id, session="s1", principal="ishan", confirmation="yes")
    store.approve(pending.id, session="s1", principal="ishan", confirmation="send draft-0001")
    assert store.consume(pending.id, SEND, session="s1", principal="ishan")


def test_overdue_approvals_are_closed_and_their_arguments_dropped(
    store: ApprovalStore, clock: Clock
) -> None:
    pending = store.request(DRAFT, risk="write", session="s1", principal="ishan")
    assert store.get(pending.id).args is not None  # type: ignore[union-attr]
    clock.now = T0 + timedelta(seconds=601)
    assert store.expire_overdue() == 1
    record = store.get(pending.id)
    assert record is not None
    assert (record.status, record.args) == ("expired", None)


def _raw_rows(db: Path) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT status, args_json FROM approvals ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()


def test_expired_requests_nobody_reopens_lose_their_arguments(tmp_path: Path) -> None:
    db = tmp_path / "approvals.sqlite"
    first = Clock()
    ApprovalStore(db, ttl_s=600, clock=first).request(
        DRAFT, risk="write", session="s1", principal="ishan"
    )
    assert b"Secret plan B." in db.read_bytes()

    later = Clock()
    later.now = T0 + timedelta(seconds=601)
    ApprovalStore(db, ttl_s=600, clock=later)  # any later command opens the store

    assert _raw_rows(db) == [("expired", None)]
    assert b"Secret plan B." not in db.read_bytes()  # secure_delete zeroes the old cell


def test_used_and_denied_requests_lose_their_arguments(tmp_path: Path, clock: Clock) -> None:
    db = tmp_path / "approvals.sqlite"
    store = ApprovalStore(db, ttl_s=600, clock=clock)
    used = store.request(DRAFT, risk="write", session="s1", principal="ishan")
    store.approve(used.id, session="s1", principal="ishan")
    store.consume(used.id, DRAFT, session="s1", principal="ishan")
    other = DRAFT.model_copy(update={"args": {**DRAFT.args, "body": "Second secret."}})
    denied = store.request(other, risk="write", session="s1", principal="ishan")
    store.deny(denied.id, session="s1", principal="ishan")

    assert _raw_rows(db) == [("consumed", None), ("denied", None)]
    raw = db.read_bytes()
    assert b"Secret plan B." not in raw
    assert b"Second secret." not in raw


class RacingStore(ApprovalStore):
    """Pauses consumers right after their checks, so both pass them before either commits."""

    barrier = threading.Barrier(2)

    def _open(self, approval_id: str, session: str, principal: str, *, expect: Any) -> Any:
        approval = super()._open(approval_id, session, principal, expect=expect)
        if expect == "approved":
            self.barrier.wait(timeout=10)
        return approval


def _race(db: Path, approval_id: str, consume: Any) -> list[Any]:
    RacingStore.barrier = threading.Barrier(2)
    stores = [RacingStore(db, ttl_s=600) for _ in range(2)]  # two "processes"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume, s, approval_id) for s in stores]
        return [f.exception() or f.result() for f in futures]


def test_concurrent_consumption_has_exactly_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "approvals.sqlite"
    approval_id = _approved(ApprovalStore(db, ttl_s=600))

    def consume(store: ApprovalStore, aid: str) -> Grant:
        return store.consume(aid, DONE, session="s1", principal="ishan")

    results = _race(db, approval_id, consume)
    grants = [r for r in results if isinstance(r, Grant)]
    refused = [r for r in results if isinstance(r, ApprovalError)]
    assert len(grants) == 1
    assert [e.reason for e in refused] == ["approval already used"]


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any) -> Any:
        with self._lock:
            self.calls += 1
        payload = json.dumps({"key": args["key"], "status": args["status"]})
        return SimpleNamespace(content=[TextContent(type="text", text=payload)], is_error=False)


def test_concurrent_approved_runs_dispatch_the_write_at_most_once(tmp_path: Path) -> None:
    db = tmp_path / "approvals.sqlite"
    approval_id = _approved(ApprovalStore(db, ttl_s=600))
    client = CountingClient()

    def run(store: ApprovalStore, aid: str) -> ActionOutcome:
        runner = ActionRunner(
            store=store,
            audit=AuditLog(),
            policy=Policy(),
            data_mode="fixture",
            clients={"jira": client},  # type: ignore[dict-item]
        )

        async def go() -> ActionOutcome:
            return await runner.execute(aid, session="s1", principal="ishan")

        return anyio.run(go)

    outcomes = _race(db, approval_id, run)
    assert sorted(o.status for o in outcomes) == ["blocked", "succeeded"]
    assert client.calls == 1


def test_store_survives_reopening(tmp_path: Path, clock: Clock) -> None:
    db = tmp_path / "approvals.sqlite"
    approval_id = _approved(ApprovalStore(db, ttl_s=600, clock=clock))
    reopened = ApprovalStore(db, ttl_s=600, clock=clock)
    assert reopened.consume(approval_id, DONE, session="s1", principal="ishan")
    with pytest.raises(ApprovalError, match="already used"):
        ApprovalStore(db, ttl_s=600, clock=clock).consume(
            approval_id, DONE, session="s1", principal="ishan"
        )


# --- audit log ----------------------------------------------------------------------------


def test_audit_log_keeps_safe_summaries_only(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path)
    audit.record("proposed", DRAFT, session="s1", principal="ishan", approval_id="apr-1")
    audit.record("proposed", DONE, session="s1", principal="ishan", approval_id="apr-2")
    audit.record(
        "failed",
        Action(source="gmail", tool="unknown_tool", args={"password": "hunter2"}),
        session="s1",
        principal="ishan",
        reason="tool_error",
    )
    raw = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [r["event"] for r in lines] == ["proposed", "proposed", "failed"]
    assert "Secret plan B." not in raw
    assert "Cutover" not in raw
    assert "priya.sharma" not in raw
    assert "hunter2" not in raw
    assert lines[0]["args"] == {
        "recipients": 1,
        "recipient_domains": ["acme.test"],
        "subject_chars": 7,
        "body_chars": 14,
        "thread_id": None,
    }
    assert lines[1]["args"] == {
        "key": "PROJ-142",
        "status": "Done",
        "assignee_set": False,
        "comment_chars": 0,
    }
    assert lines[2]["args"] == {"arg_names": ["password"]}
    assert lines[0]["action_hash"] == DRAFT.fingerprint()
    assert set(lines[0]) >= {"ts", "event", "session", "principal", "approval_id", "source", "tool"}


# --- the MCP call choke point -------------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any) -> Any:
        self.calls.append(name)
        raise AssertionError("the client must not be reached")


@pytest.mark.anyio
async def test_call_json_refuses_writes_without_a_matching_grant() -> None:
    client = FakeClient()
    with pytest.raises(PermissionDenied, match="needs an approval"):
        await call_json(client, "update_issue", DONE.args)  # type: ignore[arg-type]
    wrong = Grant(approval_id="apr-1", action_hash=SEND.fingerprint())
    with pytest.raises(PermissionDenied, match="does not match"):
        await call_json(client, "update_issue", DONE.args, grant=wrong)  # type: ignore[arg-type]
    with pytest.raises(PermissionDenied, match="unknown tool"):
        await call_json(client, "forward_all_invoices", {})  # type: ignore[arg-type]
    assert client.calls == []
