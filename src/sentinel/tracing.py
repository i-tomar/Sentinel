"""OpenTelemetry tracing, exported to Langfuse's OTLP endpoint when TRACING_BACKEND=langfuse.

Every request gets a real W3C trace id even with TRACING_BACKEND=none (spans are created but not
exported), so responses can always return `trace_id`. PII is redacted twice: when attributes are
set (`set_text`) and again by `RedactingSpanExporter` before anything leaves the process. No
secret is ever written to a span; credentials only appear in exporter HTTP headers.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, Tracer
from opentelemetry.util.types import Attributes

from sentinel import __version__
from sentinel.config import Settings
from sentinel.guards.pii import redact_text

TRACER_NAME = "sentinel"
MAX_ATTRIBUTE_CHARS = 8000


def trace_id_hex(span: Span) -> str:
    return format(span.get_span_context().trace_id, "032x")


def set_text(span: Span, key: str, value: str) -> None:
    """Set a string attribute after PII redaction and truncation."""
    span.set_attribute(key, redact_text(value)[:MAX_ATTRIBUTE_CHARS])


def _redact_attributes(attributes: Attributes) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in (attributes or {}).items():
        if isinstance(value, str):
            out[key] = redact_text(value)
        elif isinstance(value, Sequence) and value and isinstance(value[0], str):
            out[key] = tuple(redact_text(str(v)) for v in value)
        else:
            out[key] = value
    return out


class RedactingSpanExporter(SpanExporter):
    """Defense in depth: re-redacts every string attribute, event (including recorded
    exceptions' messages and stack traces) and status description before export, so nothing
    unredacted leaves the process even if a caller bypassed `set_text`."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        cleaned = [
            ReadableSpan(
                name=s.name,
                context=s.context,
                parent=s.parent,
                resource=s.resource,
                attributes=_redact_attributes(s.attributes),  # type: ignore[arg-type]
                events=[
                    Event(e.name, _redact_attributes(e.attributes), e.timestamp)  # type: ignore[arg-type]
                    for e in s.events
                ],
                links=s.links,
                kind=s.kind,
                status=Status(
                    s.status.status_code,
                    redact_text(s.status.description) if s.status.description else None,
                ),
                start_time=s.start_time,
                end_time=s.end_time,
                instrumentation_scope=s.instrumentation_scope,
            )
            for s in spans
        ]
        return self.inner.export(cleaned)

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


def langfuse_otlp_config(settings: Settings) -> tuple[str, dict[str, str]]:
    """Endpoint and auth headers for Langfuse's OTLP/HTTP trace ingestion."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        raise ValueError("Langfuse tracing needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    pair = (
        f"{settings.langfuse_public_key.get_secret_value()}:"
        f"{settings.langfuse_secret_key.get_secret_value()}"
    )
    token = base64.b64encode(pair.encode()).decode()
    endpoint = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"
    return endpoint, {"Authorization": f"Basic {token}"}


def build_tracer_provider(
    settings: Settings, exporter: SpanExporter | None = None
) -> TracerProvider:
    """`exporter` overrides the configured backend (tests pass an in-memory exporter)."""
    provider = TracerProvider(
        resource=Resource.create({"service.name": "sentinel", "service.version": __version__})
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter)))
    elif settings.tracing_backend == "langfuse":
        endpoint, headers = langfuse_otlp_config(settings)
        otlp = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=10)
        provider.add_span_processor(BatchSpanProcessor(RedactingSpanExporter(otlp)))
    elif settings.tracing_backend == "langsmith":
        raise NotImplementedError("TRACING_BACKEND=langsmith is not implemented yet; use langfuse")
    return provider


def get_tracer(provider: TracerProvider) -> Tracer:
    return provider.get_tracer(TRACER_NAME, __version__)


def generation_attributes(
    model: str, input_tokens: int, output_tokens: int
) -> Mapping[str, str | int]:
    """OTel GenAI semantic-convention attributes, which Langfuse maps to a generation."""
    return {
        "langfuse.observation.type": "generation",
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
    }
