"""Run routed tool calls with a timeout, up to N retries with exponential backoff, and record
failures instead of raising, so the agent can continue with partial evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import anyio
from mcp import Client

from sentinel.graph.state import ToolCall, ToolFailure
from sentinel.guards.redaction import redact_str
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
    def __init__(self, clients: Mapping[Source, Client], timeout_s: float) -> None:
        self.clients = clients
        self.timeout_s = timeout_s

    async def call(self, source: Source, tool: str, args: dict[str, Any]) -> Any:
        client = self.clients.get(source)
        if client is None:
            raise ToolCallError(f"no MCP connection for {source}")
        return await call_json(client, tool, args, timeout_s=self.timeout_s)


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
) -> tuple[Any, int, str | None]:
    """Return (result, attempts, error). Waits backoff_s, 2*backoff_s, ... between attempts."""
    error = "not attempted"
    for attempt in range(1, retries + 2):
        try:
            with anyio.fail_after(timeout_s):
                return await executor.call(source, tool, args), attempt, None
        except TimeoutError:
            error = f"timed out after {timeout_s:g}s"
        except PermissionDenied:
            return None, attempt, "tool not allowed"  # a refusal is final, never retried
        except ToolCallError as exc:
            error = redact_str(str(exc))  # server error text may echo record content
        if attempt <= retries:
            await sleep(backoff_s * 2 ** (attempt - 1))
    return None, retries + 1, error


class ExecutionResult:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.sq_docs: dict[str, list[str]] = {}
        self.failures: list[ToolFailure] = []


async def execute_calls(
    calls: Sequence[ToolCall],
    executor: ToolExecutor,
    known: Mapping[str, Document],
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    sleep: Sleep = anyio.sleep,
    on_call: Callable[[ToolCall, int, str | None], None] | None = None,
) -> ExecutionResult:
    out = ExecutionResult()

    async def attempt(
        source: Source, tool: str, args: dict[str, Any]
    ) -> tuple[Any, int, str | None]:
        return await call_with_retry(
            executor,
            source,
            tool,
            args,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            sleep=sleep,
        )

    def fail(call: ToolCall, tool: str, error: str, attempts: int) -> None:
        out.failures.append(
            ToolFailure(
                call_id=call.id,
                sub_question_id=call.sub_question_id,
                source=call.source,
                tool=tool,
                error=error,
                attempts=attempts,
            )
        )

    for call in calls:
        keys = out.sq_docs.setdefault(call.sub_question_id, [])
        if (call.source, call.tool) not in READ_TOOLS:
            fail(call, call.tool, "tool not allowed", 0)
            continue
        result, attempts, error = await attempt(call.source, call.tool, dict(call.args))
        if on_call:
            on_call(call, attempts, error)
        if error is not None:
            fail(call, call.tool, error, attempts)
            continue
        spec = SOURCE_TOOLS[call.source]
        if call.tool == spec.get_tool:  # direct fetch (e.g. re-reading a memory's source)
            try:
                doc = NORMALIZERS[call.source](result, datetime.now(UTC))
            except ValueError as exc:
                fail(call, call.tool, f"unexpected record ({type(exc).__name__})", attempts)
                continue
            out.documents[doc.key] = doc  # fresher than any remembered copy
            out.sq_docs[call.sub_question_id] = list(dict.fromkeys([*keys, doc.key]))
            continue
        try:
            ids = [str(r[spec.id_field]) for r in result["results"]][: call.expand]
        except (KeyError, TypeError) as exc:
            fail(call, call.tool, f"unexpected search payload ({type(exc).__name__})", attempts)
            continue
        for record_id in ids:
            key = doc_key(call.source, record_id)
            if key in known or key in out.documents:
                keys.append(key)
                continue
            raw, get_attempts, get_error = await attempt(
                call.source, spec.get_tool, {spec.get_arg: record_id}
            )
            if get_error is not None:
                fail(call, spec.get_tool, get_error, get_attempts)
                continue
            try:
                doc = NORMALIZERS[call.source](raw, datetime.now(UTC))
            except ValueError as exc:  # pydantic ValidationError is a ValueError
                fail(call, spec.get_tool, f"unexpected record ({type(exc).__name__})", 1)
                continue
            out.documents[doc.key] = doc
            keys.append(doc.key)
        out.sq_docs[call.sub_question_id] = list(dict.fromkeys(keys))
    return out
