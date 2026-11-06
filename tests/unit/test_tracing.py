from __future__ import annotations

import base64

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sentinel.config import Settings
from sentinel.tracing import (
    build_tracer_provider,
    get_tracer,
    langfuse_otlp_config,
    set_text,
    trace_id_hex,
)

pytestmark = pytest.mark.unit


def _langfuse_settings() -> Settings:
    return Settings.model_validate(
        {
            "tracing_backend": "langfuse",
            "langfuse_host": "http://127.0.0.1:3000/",
            "langfuse_public_key": "pk-lf-test",
            "langfuse_secret_key": "sk-lf-very-secret",
        }
    )


def test_langfuse_otlp_endpoint_and_auth() -> None:
    endpoint, headers = langfuse_otlp_config(_langfuse_settings())
    assert endpoint == "http://127.0.0.1:3000/api/public/otel/v1/traces"
    scheme, token = headers["Authorization"].split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(token).decode() == "pk-lf-test:sk-lf-very-secret"


def test_langfuse_provider_builds_without_network() -> None:
    provider = build_tracer_provider(_langfuse_settings())
    provider.shutdown()


def test_langsmith_backend_is_not_silently_ignored() -> None:
    with pytest.raises(NotImplementedError, match="langsmith"):
        build_tracer_provider(
            Settings.model_validate({"tracing_backend": "langsmith", "langsmith_api_key": "k"})
        )


def test_none_backend_still_issues_real_trace_ids() -> None:
    tracer = get_tracer(build_tracer_provider(Settings()))
    with tracer.start_as_current_span("x") as span:
        trace_id = trace_id_hex(span)
    assert len(trace_id) == 32
    assert int(trace_id, 16) != 0


def test_exported_spans_are_redacted_and_secret_free() -> None:
    exporter = InMemorySpanExporter()
    settings = _langfuse_settings()
    tracer = get_tracer(build_tracer_provider(settings, exporter=exporter))
    with tracer.start_as_current_span("root") as span:
        set_text(span, "input", "Call Sofia at +91 98200 11223 or sofia.alvarez@acme.test")
        span.set_attribute("raw", "PAN ABCPS1234K")  # bypasses set_text: exporter must catch it
        span.add_event("note", {"detail": "Aadhaar 2345 6789 0123"})
    [exported] = exporter.get_finished_spans()
    dumped = exported.to_json()
    for leaked in ("+91 98200 11223", "sofia.alvarez@acme.test", "ABCPS1234K", "2345 6789 0123"):
        assert leaked not in dumped
    assert "<PHONE_IN>" in dumped
    assert "<PAN>" in dumped
    assert "sk-lf-very-secret" not in dumped
    assert "pk-lf-test" not in dumped
