"""Every fixture document, and every sample PII kind, through every output-producing path.

Paths: the redacted record that enters agent state, its index chunks, evidence and citation
rendering, the LLM prompt, the (mock) model's answer after the output guard, trace export
(attributes, events, status and exception records, through `RedactingSpanExporter`),
long-term memory storage, and the audit log. No raw PII value may survive any of them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from sentinel.config import Settings
from sentinel.graph.nodes.answerer import build_messages
from sentinel.graph.nodes.output_guard import guard_output
from sentinel.graph.nodes.retrieve import assemble_evidence
from sentinel.graph.state import SubQuestion
from sentinel.guards.pii import find_pii
from sentinel.guards.redaction import redact_chunk, redact_document
from sentinel.llm import MockLLM
from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.permissions.actions import Action
from sentinel.permissions.audit import AuditLog
from sentinel.retrieval.chunking import chunk_document
from sentinel.retrieval.models import Document
from sentinel.retrieval.normalize import gmail_document, jira_document, notion_document
from sentinel.tracing import build_tracer_provider, get_tracer
from servers.fixtures.generate import PII_SEEDS, build_dataset
from tests.guard_samples import PII_SAMPLES

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 2, tzinfo=UTC)
DATASET = build_dataset()
DOCS: dict[str, Document] = {
    d.key: d
    for d in (
        *(notion_document(p.model_dump(mode="json"), NOW) for p in DATASET.notion),
        *(jira_document(i.model_dump(mode="json"), NOW) for i in DATASET.jira),
        *(gmail_document(t.model_dump(mode="json"), NOW) for t in DATASET.gmail),
    )
}


def _raw_values(doc: Document) -> set[str]:
    fields = [doc.title, doc.text, doc.author or "", *doc.metadata.values()]
    found = {text[f.start : f.end] for text in fields for f in find_pii(text)}
    seeds = {s.value for s in PII_SEEDS if f"{s.source}:{s.record_id}" == doc.key}
    return found | seeds


def _outputs(doc: Document) -> list[str]:
    """Run `doc` through every output path; return everything that would leave the code."""
    out: list[str] = []
    clean = redact_document(doc)
    out += [clean.model_dump_json()]
    out += [redact_chunk(c).model_dump_json() for c in chunk_document(doc)]

    plan = [SubQuestion(id="sq1", text="q", target_sources=[doc.source])]
    evidence = assemble_evidence("q", plan, {doc.key: clean}, {"sq1": [doc.key]}, [], [])
    out += [e.model_dump_json() for e in evidence]
    out += [e.citation.render() for e in evidence]
    messages = build_messages("What does this record say?", evidence)
    out += [str(m.content) for m in messages]
    reply = str(MockLLM().invoke(messages).content)
    answer, citations, _ = guard_output(reply, [e.citation for e in evidence])
    out += [answer, *(c.model_dump_json() for c in citations)]

    exporter = InMemorySpanExporter()
    tracer = get_tracer(build_tracer_provider(Settings(), exporter=exporter))
    with tracer.start_as_current_span("path") as span:
        span.set_attribute("raw.text", doc.text)  # bypasses set_text on purpose
        span.set_attribute("raw.list", [doc.title, doc.text])
        span.add_event("raw", {"text": doc.text})
        span.record_exception(ValueError(doc.text[:300]))
        span.set_status(Status(StatusCode.ERROR, doc.text[:300]))
    out += [s.to_json() for s in exporter.get_finished_spans()]

    memory = SQLiteMemoryBackend(":memory:")
    stored = memory.put(
        MemoryItem(key="k", value=doc.text[:500] or "x", source="user", updated_at=NOW)
    )
    memory.mark_stale("k", (doc.text[:300] or "x"))
    out += [stored.model_dump_json(), json.dumps([i.model_dump(mode="json") for i in memory.all()])]

    audit = AuditLog()
    action = Action(
        source="gmail",
        tool="create_draft",
        args={"to": [doc.author or "a@acme.test"], "subject": doc.title, "body": doc.text},
    )
    out += [audit.record("proposed", action, session="s", principal="u").model_dump_json()]
    return out


@pytest.mark.parametrize("key", sorted(DOCS))
def test_no_raw_pii_from_any_fixture_document_leaves_any_path(key: str) -> None:
    doc = DOCS[key]
    outputs = _outputs(doc)
    for value in _raw_values(doc):
        leaked = [o[:80] for o in outputs if value in o]
        assert not leaked, f"{key}: {value!r} leaked"


@pytest.mark.parametrize("kind", sorted(PII_SAMPLES))
def test_each_pii_kind_is_removed_from_every_path(kind: str) -> None:
    value = PII_SAMPLES[kind]
    doc = Document(
        source="gmail",
        id="thr-9999",
        url="https://mail.fixture.acme.test/thread/thr-9999",
        title=f"Re: details {value}",
        text=f"From: ops@acme.test\n\nHello, my details are {value}. Thanks.",
        updated_at=NOW,
        fetched_at=NOW,
        author=value if kind == "EMAIL" else "ops@acme.test",
        metadata={"note": f"see {value}"},
    )
    outputs = _outputs(doc)
    assert not [o[:80] for o in outputs if value in o], kind


def test_citation_ids_and_urls_survive_redaction() -> None:
    doc = DOCS["gmail:thr-0037"]  # a phone number in a signature
    clean = redact_document(doc)
    assert (clean.id, clean.url, clean.source) == (doc.id, doc.url, doc.source)
    assert "+91 98200 11223" not in clean.text
    assert "<PHONE_IN>" in clean.text
    plan = [SubQuestion(id="sq1", text="q", target_sources=["gmail"])]
    [evidence] = assemble_evidence("q", plan, {doc.key: clean}, {"sq1": [doc.key]}, [], [])
    assert evidence.citation.key == "gmail:thr-0037"
    assert evidence.citation.url == doc.url
    answer, citations, kinds = guard_output("Call +91 98200 11223 [1].", [evidence.citation])
    assert answer == "Call <PHONE_IN> [1]."
    assert kinds == ["PHONE_IN"]
    assert citations[0].url == doc.url
