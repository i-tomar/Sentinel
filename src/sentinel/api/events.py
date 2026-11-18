"""Build stage events only from spans of nodes that actually ran."""

from __future__ import annotations

from datetime import UTC, datetime

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sentinel.api.models import StageEvent, StageStatus
from sentinel.graph.build import AskResult

_NODE_PREFIX = "node."


def _ns_to_dt(ns: int | None) -> datetime | None:
    if ns is None:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)


def _duration_ms(span: ReadableSpan) -> int | None:
    if span.start_time is None or span.end_time is None:
        return None
    return max(0, int((span.end_time - span.start_time) / 1_000_000))


def _span_status(span: ReadableSpan) -> StageStatus:
    code = span.status.status_code.name if span.status else "UNSET"
    if code == "ERROR":
        return "failed"
    if span.end_time is not None:
        return "completed"
    return "started"


def _stage_status(stage: str, result: AskResult, span: ReadableSpan) -> StageStatus:
    """Span outcome, plus result-driven failures for citation/output stages."""
    if stage == "citation_check" and not result.citation_report.ok:
        return "failed"
    if stage == "output_guard" and result.output_redactions:
        return "failed"
    return _span_status(span)


def _summary(stage: str, result: AskResult) -> str:
    check = result.input_check
    if stage == "input_guard":
        if check is not None and check.blocked:
            rules = ", ".join(check.rules) or "classifier"
            return f"blocked ({rules})"
        return "ok"
    if stage == "retrieved_guard":
        if result.guard_notes:
            return f"neutralized {len(result.guard_notes)} record(s)"
        return "ok"
    if stage == "citation_check":
        report = result.citation_report
        if report.ok:
            return "ok"
        parts: list[str] = []
        if report.unknown:
            parts.append(f"unknown markers {report.unknown}")
        if report.uncited_sentences:
            parts.append(f"{len(report.uncited_sentences)} uncited sentence(s)")
        if report.unsupported:
            parts.append(f"{len(report.unsupported)} unsupported claim(s)")
        if report.unverified_memory_claims:
            parts.append(f"{len(report.unverified_memory_claims)} unverified memory claim(s)")
        return "; ".join(parts) or "failed"
    if stage == "output_guard":
        if result.output_redactions:
            return "redacted: " + ", ".join(result.output_redactions)
        return "ok"
    if stage == "answer":
        if result.insufficient_evidence:
            return "insufficient evidence"
        return "answered"
    if stage == "permission_gate":
        if result.actions:
            return f"{len(result.actions)} action(s) proposed"
        return "no write actions"
    if stage == "execute_tools":
        return f"{len(result.tool_calls)} tool call(s)"
    if stage == "retrieve":
        return f"{len(result.evidence_keys)} evidence key(s)"
    return "ran"


def events_from_spans(exporter: InMemorySpanExporter, result: AskResult) -> list[StageEvent]:
    """One event per `node.*` span, ordered by start time. Empty if nothing ran."""
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith(_NODE_PREFIX)]
    spans.sort(key=lambda s: s.start_time or 0)
    events: list[StageEvent] = []
    for span in spans:
        stage = span.name.removeprefix(_NODE_PREFIX)
        events.append(
            StageEvent(
                stage=stage,
                status=_stage_status(stage, result, span),
                started_at=_ns_to_dt(span.start_time),
                completed_at=_ns_to_dt(span.end_time),
                duration_ms=_duration_ms(span),
                summary=_summary(stage, result),
            )
        )
    return events
