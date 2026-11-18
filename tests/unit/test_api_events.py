"""Stage event status reflects citation/output failures."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sentinel import __version__
from sentinel.api.events import events_from_spans
from sentinel.graph.build import AskResult
from sentinel.graph.state import CitationReport, TokenUsage
from sentinel.tracing import RedactingSpanExporter, get_tracer

pytestmark = pytest.mark.unit


def _result(**overrides: object) -> AskResult:
    data: dict[str, object] = {
        "question": "q",
        "answer": "a",
        "citations": [],
        "insufficient_evidence": False,
        "citation_report": CitationReport(),
        "searched_sources": [],
        "evidence_keys": [],
        "plan": [],
        "hops": 0,
        "unanswered": [],
        "tool_calls": [],
        "failures": [],
        "stale_keys": [],
        "errors": [],
        "guard_notes": [],
        "input_check": None,
        "thread_id": None,
        "turn": 1,
        "memory_notes": [],
        "memory_writes": [],
        "actions": [],
        "output_redactions": [],
        "session_id": "s",
        "trace_id": "0" * 32,
        "latency_ms": 1,
        "token_usage": TokenUsage(),
    }
    data.update(overrides)
    return AskResult.model_validate(data)


def _span_for(stage: str) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "sentinel", "service.version": __version__})
    )
    provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter)))
    tracer = get_tracer(provider)
    with tracer.start_as_current_span(f"node.{stage}"):
        pass
    return exporter


def test_citation_failed_marks_citation_check_event_failed() -> None:
    exporter = _span_for("citation_check")
    result = _result(citation_report=CitationReport(unknown=[42]))
    events = events_from_spans(exporter, result)
    assert len(events) == 1
    assert events[0].stage == "citation_check"
    assert events[0].status == "failed"
    assert "unknown" in events[0].summary


def test_output_redacted_marks_output_guard_event_failed() -> None:
    exporter = _span_for("output_guard")
    result = _result(output_redactions=["EMAIL_ADDRESS"])
    events = events_from_spans(exporter, result)
    assert len(events) == 1
    assert events[0].stage == "output_guard"
    assert events[0].status == "failed"
    assert "EMAIL_ADDRESS" in events[0].summary
