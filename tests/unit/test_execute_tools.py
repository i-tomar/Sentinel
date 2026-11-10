from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio
import pytest

from sentinel.graph.nodes.execute_tools import call_with_retry, execute_calls
from sentinel.graph.state import ToolCall
from sentinel.mcp_clients.tools import ToolCallError
from sentinel.retrieval.models import Document, Source

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

T = datetime(2026, 11, 2, tzinfo=UTC)
PAGE = {
    "id": "p1",
    "title": "Plan",
    "url": "https://n.test/p1",
    "owner": "a@acme.test",
    "updated_at": T.isoformat(),
    "tags": [],
    "body": "PROJ-120",
}


class FakeExecutor:
    def __init__(self, fail_times: int = 0, hang: bool = False, payload: Any = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_times = fail_times
        self.hang = hang
        self.payload = payload

    async def call(self, source: Source, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((source, tool, args))
        if self.hang:
            await anyio.sleep(10)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ToolCallError(f"{tool} failed: 500 Internal Server Error", kind="server_error")
        if tool == "search_pages":
            return self.payload if self.payload is not None else {"results": [{"id": "p1"}]}
        return PAGE


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _call(tool: str = "search_pages", source: Source = "notion") -> ToolCall:
    return ToolCall(
        id="h1-c1",
        sub_question_id="sq1",
        hop=1,
        source=source,
        tool=tool,
        args={"query": "plan", "limit": 5},
        expand=3,
    )


async def test_retry_succeeds_after_transient_errors_with_backoff() -> None:
    ex, sleeps = FakeExecutor(fail_times=2), Sleeps()
    result, attempts, error = await call_with_retry(
        ex,
        "notion",
        "get_page",
        {"page_id": "p1"},
        timeout_s=1,
        retries=2,
        backoff_s=0.5,
        sleep=sleeps,
    )
    assert (attempts, error) == (3, None)
    assert result == PAGE
    assert sleeps.waits == [0.5, 1.0]


async def test_retry_gives_up_after_limit() -> None:
    ex, sleeps = FakeExecutor(fail_times=99), Sleeps()
    result, attempts, error = await call_with_retry(
        ex, "notion", "get_page", {}, timeout_s=1, retries=2, backoff_s=0.25, sleep=sleeps
    )
    assert result is None
    assert attempts == 3
    assert error is not None
    assert error.kind == "server_error"
    assert len(ex.calls) == 3
    assert sleeps.waits == [0.25, 0.5]


async def test_timeout_is_enforced_per_attempt() -> None:
    ex, sleeps = FakeExecutor(hang=True), Sleeps()
    with anyio.fail_after(5):
        _, attempts, error = await call_with_retry(
            ex, "notion", "get_page", {}, timeout_s=0.05, retries=1, backoff_s=0, sleep=sleeps
        )
    assert attempts == 2
    assert error is not None
    assert (error.kind, error.description) == ("timeout", "timed out after 0.05s")


async def test_search_then_expand_and_reuse_known_docs() -> None:
    ex = FakeExecutor()
    out = await execute_calls([_call()], ex, {}, timeout_s=1, retries=0, backoff_s=0)
    assert [t for _, t, _ in ex.calls] == ["search_pages", "get_page"]
    assert out.sq_docs == {"sq1": ["notion:p1"]}
    assert out.documents["notion:p1"].text == "PROJ-120"

    known: dict[str, Document] = dict(out.documents)
    ex2 = FakeExecutor()
    out2 = await execute_calls([_call()], ex2, known, timeout_s=1, retries=0, backoff_s=0)
    assert [t for _, t, _ in ex2.calls] == ["search_pages"]  # no refetch
    assert out2.sq_docs == {"sq1": ["notion:p1"]}


async def test_disallowed_tool_is_never_called() -> None:
    ex = FakeExecutor()
    out = await execute_calls(
        [_call(tool="update_issue", source="jira")], ex, {}, timeout_s=1, retries=0, backoff_s=0
    )
    assert ex.calls == []
    assert [(f.tool, f.error, f.attempts) for f in out.failures] == [
        ("update_issue", "tool not allowed", 0)
    ]


async def test_unexpected_search_payload_is_a_failure() -> None:
    ex = FakeExecutor(payload={"rows": []})
    out = await execute_calls([_call()], ex, {}, timeout_s=1, retries=0, backoff_s=0)
    assert out.failures[0].kind == "invalid_response"
    assert out.failures[0].error == "invalid response (malformed or unexpected data)"
