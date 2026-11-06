"""Opt-in: export a real `ask` trace to a running Langfuse (e.g. `make up`) over OTLP/HTTP.

Set SENTINEL_TEST_LANGFUSE_INFRA_ENV to the absolute path of `.env.infra`. Asserts Langfuse
accepted every export batch (HTTP 2xx); nothing is read back.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from qdrant_client import QdrantClient

from sentinel.config import Settings
from sentinel.graph.build import Deps, ask
from sentinel.infra import app_env_lines, read_infra_env
from sentinel.llm import MockLLM
from sentinel.retrieval.chunking import chunk_document
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.models import Document
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import build_tracer_provider, get_tracer, langfuse_otlp_config

INFRA_ENV = os.environ.get("SENTINEL_TEST_LANGFUSE_INFRA_ENV")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INFRA_ENV, reason="set SENTINEL_TEST_LANGFUSE_INFRA_ENV"),
]


class _Recording(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner
        self.results: list[SpanExportResult] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = self.inner.export(spans)
        self.results.append(result)
        return result

    def shutdown(self) -> None:
        self.inner.shutdown()


def test_ask_trace_is_accepted_by_langfuse() -> None:
    assert INFRA_ENV
    lines = app_env_lines(read_infra_env(Path(INFRA_ENV)), host="http://127.0.0.1:3000")
    settings = Settings.model_validate(
        {k.lower(): v for k, v in (line.split("=", 1) for line in lines)}
    )
    endpoint, headers = langfuse_otlp_config(settings)
    recorder = _Recording(OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10))
    provider = build_tracer_provider(settings, exporter=recorder)

    store = ChunkStore(QdrantClient(location=":memory:"), "lf", HashEmbedder())
    store.ensure_collection()
    doc = Document(
        source="notion",
        id="auth-migration-v2",
        url="https://notion.fixture.acme.test/auth-migration-v2",
        title="Q3 Auth Migration Plan (v2)",
        text="Owner: Priya Sharma. Call +91 98450 10001. Cutover date: 2026-11-20.",
        updated_at=__import__("datetime").datetime(2026, 10, 27),
        fetched_at=__import__("datetime").datetime(2026, 11, 2),
    )
    store.replace_document(chunk_document(doc))
    result = ask(
        "Who owns the auth migration plan?",
        Deps(store=store, llm=MockLLM(), tracer=get_tracer(provider)),
    )
    provider.shutdown()
    assert result.citations
    assert recorder.results
    assert all(r == SpanExportResult.SUCCESS for r in recorder.results)
    print(f"\nexported trace_id={result.trace_id}")
