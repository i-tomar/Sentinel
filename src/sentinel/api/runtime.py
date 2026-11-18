"""Fixture/mock demo runtime. Always mock LLM + fixture data; never honors live overrides."""

from __future__ import annotations

import uuid
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any, Literal

from mcp import Client
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from sentinel import __version__
from sentinel.api.events import events_from_spans
from sentinel.api.models import RunStatus, StageEvent, run_status_from_result
from sentinel.api.store import RunStore, StoredRun
from sentinel.graph.build import AskResult, Deps, ask_async
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.llm import MockLLM
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.ingest import ingest
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import RedactingSpanExporter, get_tracer
from servers.fixtures import gmail_mcp, jira_mcp, notion_mcp

# Demo safety: fixed regardless of process env or client payload.
DEMO_LLM: Literal["mock"] = "mock"
DEMO_DATA_MODE: Literal["fixture"] = "fixture"


class DemoRuntime:
    """Shared in-memory index + fixture MCP tools for the API process."""

    def __init__(self) -> None:
        self.store = RunStore()
        self._stack: AsyncExitStack | None = None
        self._chunk_store: ChunkStore | None = None
        self._clients: dict[Source, Client] | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def startup(self) -> None:
        if self._ready:
            return
        stack = AsyncExitStack()
        await stack.__aenter__()
        clients: dict[Source, Client] = {
            "notion": await stack.enter_async_context(Client(notion_mcp.create_server())),
            "jira": await stack.enter_async_context(Client(jira_mcp.create_server())),
            "gmail": await stack.enter_async_context(Client(gmail_mcp.create_server())),
        }
        chunk_store = ChunkStore(QdrantClient(location=":memory:"), "api-demo", HashEmbedder())
        chunk_store.ensure_collection()
        reports = await ingest(clients, chunk_store, fetched_at=datetime(2026, 11, 2, tzinfo=UTC))
        if any(r.error is not None for r in reports):
            await stack.aclose()
            raise RuntimeError("fixture ingest failed")
        self._stack = stack
        self._clients = clients
        self._chunk_store = chunk_store
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False
        self._clients = None
        self._chunk_store = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    def _deps(self, exporter: InMemorySpanExporter) -> Deps:
        if self._chunk_store is None or self._clients is None:
            raise RuntimeError("runtime not started")
        # In-memory tracer only — no BaseSettings / env / .env (avoids loading API keys).
        provider = TracerProvider(
            resource=Resource.create({"service.name": "sentinel", "service.version": __version__})
        )
        provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter)))
        tracer = get_tracer(provider)
        return Deps(
            store=self._chunk_store,
            llm=MockLLM(),
            tracer=tracer,
            tools=MCPToolExecutor(self._clients, timeout_s=10.0),
            data_mode=DEMO_DATA_MODE,
            memory_auto_approve=False,
            principal="api-demo",
        )

    async def run_ask(self, question: str, thread_id: str | None = None) -> StoredRun:
        if not self._ready:
            raise RuntimeError("runtime not started")
        exporter = InMemorySpanExporter()
        deps = self._deps(exporter)
        # Single-shot demo: no checkpointer. thread_id is accepted for forward
        # compatibility but does not enable live or cross-request memory.
        _ = thread_id
        result = await ask_async(question, deps)
        events = events_from_spans(exporter, result)
        status = run_status_from_result(
            blocked=bool(result.input_check and result.input_check.blocked),
            citation_ok=result.citation_report.ok,
            output_redactions=list(result.output_redactions),
        )
        run = StoredRun(
            run_id=uuid.uuid4().hex,
            status=status,
            result=result,
            events=events,
        )
        self.store.put(run)
        return run


def result_payload(run: StoredRun) -> dict[str, Any]:
    """Shape stored AskResult for GET /api/runs/{id} (no tool args / secrets)."""
    from sentinel.api.models import DataProvenance, RunResultResponse, action_records_public

    r: AskResult = run.result
    body = RunResultResponse(
        run_id=run.run_id,
        status=run.status,
        question=r.question,
        answer=r.answer,
        citations=list(r.citations),
        input_check=r.input_check,
        guard_notes=list(r.guard_notes),
        citation_report=r.citation_report,
        actions=action_records_public(list(r.actions)),
        output_redactions=list(r.output_redactions),
        session_id=r.session_id,
        trace_id=r.trace_id,
        latency_ms=r.latency_ms,
        thread_id=r.thread_id,
        data_provenance=DataProvenance(),
    )
    return body.model_dump(mode="json")


def events_payload(run: StoredRun) -> dict[str, Any]:
    from sentinel.api.models import RunEventsResponse

    return RunEventsResponse(
        run_id=run.run_id,
        status=run.status,
        events=list(run.events),
    ).model_dump(mode="json")


def health_payload() -> dict[str, Any]:
    from sentinel.api.models import HealthResponse

    return HealthResponse(checked_at=datetime.now(UTC)).model_dump(mode="json")


__all__ = [
    "DEMO_DATA_MODE",
    "DEMO_LLM",
    "DemoRuntime",
    "RunStatus",
    "StageEvent",
    "events_payload",
    "health_payload",
    "result_payload",
]
