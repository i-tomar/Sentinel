"""Run routed read calls with a timeout and bounded retries, and record failures instead of
raising, so the agent can continue with partial evidence.

- Only transient errors (timeout, rate limit, 5xx) are retried, with exponential backoff.
  Invalid arguments, not-found, malformed data, a dead connection and permission refusals are
  final on the first attempt.
- A source that is down (transient error after its retries, or unavailable) is skipped for
  the rest of the turn, and no call starts after the turn's tool time budget has run out.
- What is recorded about a failure is its kind and a fixed description, never the server's
  error text, which is untrusted (it can echo input, record content or instructions).
- Malformed results are dropped one by one; whatever is valid is still used, and goes
  through the same redaction and retrieved-content guard as any other record.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import anyio
from mcp import Client

from sentinel.graph.state import ToolCall, ToolFailure
from sentinel.mcp_clients.failures import RETRYABLE, SOURCE_DOWN, FailureKind, describe
from sentinel.mcp_clients.tools import (
    NORMALIZERS,
    READ_TOOLS,
    SOURCE_TOOLS,
    ToolCallError,
    call_json,
)
from sentinel.permissions.actions import PermissionDenied
from sentinel.retrieval.models import Document, Source, doc_key

Sleep = Callable[[float], Awaitable[None]]


class ToolExecutor(Protocol):
    async def call(self, source: Source, tool: str, args: dict[str, Any]) -> Any: ...


class MCPToolExecutor:
    def __init__(
        self,
        clients: Mapping[Source, Client],
        timeout_s: float,
        unavailable: Mapping[Source, str] | None = None,
    ) -> None:
        """`unavailable`: sources whose server could not be started (see pool)."""
        self.clients = clients
        self.timeout_s = timeout_s
        self.unavailable = dict(unavailable or {})

    async def call(self, source: Source, tool: str, args: dict[str, Any]) -> Any:
        client = self.clients.get(source)
        if client is None:
            raise ToolCallError(f"no MCP connection for {source}", kind="unavailable")
        return await call_json(client, tool, args, timeout_s=self.timeout_s)


@dataclass(frozen=True)
class CallFailure:
    kind: FailureKind
    description: str


async def call_with_retry(
    executor: ToolExecutor,
    source: Source,
    tool: str,
    args: dict[str, Any],
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    sleep: Sleep = anyio.sleep,
    deadline: float | None = None,
) -> tuple[Any, int, CallFailure | None]:
    """Return (result, attempts, failure). Retries transient errors only, waiting backoff_s,
    2*backoff_s, ... in between, and never starts an attempt after `deadline` (monotonic)."""
    failure = CallFailure("tool_error", describe("tool_error"))
    attempt = 0
    for attempt in range(1, retries + 2):
        try:
            with anyio.fail_after(timeout_s):
                return await executor.call(source, tool, args), attempt, None
        except TimeoutError:
            failure = CallFailure("timeout", describe("timeout", timeout_s))
        except PermissionDenied:
            return None, attempt, CallFailure("permission_denied", describe("permission_denied"))
        except ToolCallError as exc:
            failure = CallFailure(
                exc.kind, describe(exc.kind, timeout_s if exc.kind == "timeout" else None)
            )
        if failure.kind not in RETRYABLE or attempt > retries:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        await sleep(backoff_s * 2 ** (attempt - 1))
    return None, attempt, failure


class ExecutionResult:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.sq_docs: dict[str, list[str]] = {}
        self.failures: list[ToolFailure] = []


def _normalize(source: Source, raw: Any) -> Document | None:
    try:
        return NORMALIZERS[source](raw, datetime.now(UTC))
    except (ValueError, KeyError, TypeError, AttributeError):  # ValidationError is a ValueError
        return None


def _hit_ids(payload: Any, id_field: str) -> tuple[list[str], bool] | None:
    """(ids of well-formed hits, whether any hit was malformed), or None if the payload itself
    is not a search result."""
    hits = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return None
    ids = [
        str(h[id_field])
        for h in hits
        if isinstance(h, dict) and isinstance(h.get(id_field), str | int) and str(h[id_field])
    ]
    return ids, len(ids) < len(hits)


async def execute_calls(
    calls: Sequence[ToolCall],
    executor: ToolExecutor,
    known: Mapping[str, Document],
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    sleep: Sleep = anyio.sleep,
    on_call: Callable[[ToolCall, int, CallFailure | None], None] | None = None,
    down: set[str] | None = None,
    deadline: float | None = None,
) -> ExecutionResult:
    """`down`: sources that already failed this turn; they are not called again, and sources
    that fail during this batch are added to a local copy. `deadline`: monotonic time after
    which no call starts."""
    out = ExecutionResult()
    down = set(down or ())

    def fail(call: ToolCall, tool: str, kind: FailureKind, attempts: int, text: str = "") -> None:
        out.failures.append(
            ToolFailure(
                call_id=call.id,
                sub_question_id=call.sub_question_id,
                source=call.source,
                tool=tool,
                error=text or describe(kind),
                attempts=attempts,
                kind=kind,
            )
        )

    def blocked(call: ToolCall, tool: str) -> bool:
        """Record and return True when the call must not start."""
        if call.source in down:
            fail(call, tool, "skipped", 0)
            return True
        if deadline is not None and time.monotonic() >= deadline:
            fail(call, tool, "budget_exhausted", 0)
            return True
        return False

    async def attempt(
        call: ToolCall, tool: str, args: dict[str, Any]
    ) -> tuple[Any, int, CallFailure | None]:
        result, attempts, failure = await call_with_retry(
            executor,
            call.source,
            tool,
            args,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            sleep=sleep,
            deadline=deadline,
        )
        if failure is not None:
            fail(call, tool, failure.kind, attempts, failure.description)
            if failure.kind in SOURCE_DOWN:
                down.add(call.source)
        return result, attempts, failure

    for call in calls:
        keys = out.sq_docs.setdefault(call.sub_question_id, [])
        if (call.source, call.tool) not in READ_TOOLS:
            fail(call, call.tool, "permission_denied", 0)
            continue
        if blocked(call, call.tool):
            continue
        result, attempts, failure = await attempt(call, call.tool, dict(call.args))
        if on_call:
            on_call(call, attempts, failure)
        if failure is not None:
            continue
        spec = SOURCE_TOOLS[call.source]
        if call.tool == spec.get_tool:  # direct fetch (e.g. re-reading a memory's source)
            doc = _normalize(call.source, result)
            if doc is None:
                fail(call, call.tool, "invalid_response", attempts)
                continue
            out.documents[doc.key] = doc  # fresher than any remembered copy
            out.sq_docs[call.sub_question_id] = list(dict.fromkeys([*keys, doc.key]))
            continue
        hits = _hit_ids(result, spec.id_field)
        if hits is None:
            fail(call, call.tool, "invalid_response", attempts)
            continue
        ids, some_malformed = hits
        if some_malformed:
            fail(call, call.tool, "invalid_response", attempts)  # the valid hits still count
        for record_id in ids[: call.expand]:
            key = doc_key(call.source, record_id)
            if key in known or key in out.documents:
                keys.append(key)
                continue
            if blocked(call, spec.get_tool):
                break
            raw, _, get_failure = await attempt(call, spec.get_tool, {spec.get_arg: record_id})
            if get_failure is not None:
                continue
            doc = _normalize(call.source, raw)
            if doc is None:
                fail(call, spec.get_tool, "invalid_response", 1)
                continue
            out.documents[doc.key] = doc
            keys.append(doc.key)
        out.sq_docs[call.sub_question_id] = list(dict.fromkeys(keys))
    return out
