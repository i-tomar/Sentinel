"""The step (e) graph over really ingested fixture data (embedded Qdrant, HashEmbedder)."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime

import anyio
import httpx2
import pytest
from mcp import Client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from sentinel.config import Settings
from sentinel.graph.build import AskResult, Deps, ask
from sentinel.graph.nodes.retrieve import relevant_chunks
from sentinel.llm import INSUFFICIENT_PREFIX, MockLLM, make_chat_model
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.ingest import ingest
from sentinel.retrieval.models import SOURCES, Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import build_tracer_provider, get_tracer
from servers.fixtures import gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.generate import PII_SEEDS

pytestmark = pytest.mark.integration

OWNER_Q = "Who owns the Q3 auth migration plan v2?"
SECRETS = ("sk-llm-secret-123", "sk-lf-secret-456", "pk-lf-public-789")


@pytest.fixture(scope="module")
def store() -> ChunkStore:
    s = ChunkStore(QdrantClient(location=":memory:"), "ask", HashEmbedder())

    async def load() -> None:
        async with AsyncExitStack() as stack:
            clients: dict[Source, Client] = {
                "notion": await stack.enter_async_context(Client(notion_mcp.create_server())),
                "jira": await stack.enter_async_context(Client(jira_mcp.create_server())),
                "gmail": await stack.enter_async_context(Client(gmail_mcp.create_server())),
            }
            reports = await ingest(clients, s, fetched_at=datetime(2026, 11, 2, tzinfo=UTC))
            assert all(r.error is None for r in reports)

    anyio.run(load)
    return s


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _settings(**overrides: str) -> Settings:
    base = {
        "tracing_backend": "langfuse",
        "langfuse_public_key": SECRETS[2],
        "langfuse_secret_key": SECRETS[1],
    }
    return Settings.model_validate({**base, **overrides})


def _deps(store: ChunkStore, exporter: InMemorySpanExporter, llm: object = None) -> Deps:
    tracer = get_tracer(build_tracer_provider(_settings(), exporter=exporter))
    return Deps(store=store, llm=llm or MockLLM(), tracer=tracer)  # type: ignore[arg-type]


def _scripted(store: ChunkStore, exporter: InMemorySpanExporter, q: str, reply: str) -> MockLLM:
    """Dry-run once to capture the exact answer prompt, then script the reply for it."""
    dry = MockLLM()
    ask(q, _deps(store, exporter, dry))
    return MockLLM(scripted={dry.calls[-1]: reply})


def _assert_citations_resolve(result: AskResult) -> None:
    for citation in result.citations:
        assert citation.key in result.evidence_keys
        assert citation.url.startswith("https://")
        assert f"[{citation.n}]" in result.answer


# --- cited answers ------------------------------------------------------------------------


def test_cited_answer_from_retrieved_evidence(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    result = ask(OWNER_Q, _deps(store, exporter))
    assert not result.insufficient_evidence
    assert result.citations
    assert "notion:auth-migration-v2" in {c.key for c in result.citations}
    assert result.citation_report.ok, result.citation_report  # cited, known and supported
    _assert_citations_resolve(result)
    assert len(result.trace_id) == 32


def test_unknown_citation_numbers_never_reach_the_answer(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    script = "Priya Sharma owns the auth migration plan [1]. It was approved by the board [42]."
    llm = _scripted(store, exporter, OWNER_Q, script)
    result = ask(OWNER_Q, _deps(store, exporter, llm))
    assert llm.calls  # the scripted reply was actually used
    assert result.citation_report.unknown == [42]
    assert "[42]" not in result.answer
    assert "Removed citation number(s) 42" in result.answer
    assert [c.n for c in result.citations] == [1]
    _assert_citations_resolve(result)


def test_insufficient_evidence_does_not_call_the_llm(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    llm = MockLLM()
    result = ask("What is the capital of France?", _deps(store, exporter, llm))
    assert result.insufficient_evidence
    assert result.answer.startswith(INSUFFICIENT_PREFIX)
    assert "notion, jira, gmail" in result.answer
    assert result.citations == []
    assert result.evidence_keys == []
    assert llm.calls == []
    assert len(result.trace_id) == 32


def test_llm_saying_insufficient_is_respected(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    q = "When will PROJ-142 be released to customers?"
    assert relevant_chunks(q, store.search(q, 6, sources=list(SOURCES)))  # related, no date
    script = f"{INSUFFICIENT_PREFIX}: the sources do not give a release date for PROJ-142."
    llm = _scripted(store, exporter, q, script)
    result = ask(q, _deps(store, exporter, llm))
    assert result.insufficient_evidence
    assert result.citations == []


# --- tracing ------------------------------------------------------------------------------


def test_spans_share_the_returned_trace_id(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    result = ask(OWNER_Q, _deps(store, exporter))
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {
        "sentinel.ask",
        "node.retrieve",
        "node.answer",
        "node.citation_check",
        "llm.generate",
    } <= names
    assert {format(s.context.trace_id, "032x") for s in spans} == {result.trace_id}
    generation = next(s for s in spans if s.name == "llm.generate")
    assert generation.attributes is not None
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert generation.attributes["gen_ai.request.model"] == "sentinel-mock"


@pytest.mark.parametrize(
    "question",
    [
        "What PAN and Aadhaar did Kabir send for the relocation reimbursement documents?",
        "What is the phone number of Sofia Alvarez in Customer Success?",
        "Initech duplicate invoice customer contact Ravi Kumar phone",
    ],
)
def test_traces_contain_no_seeded_pii_or_secrets(
    store: ChunkStore, exporter: InMemorySpanExporter, question: str
) -> None:
    result = ask(question, _deps(store, exporter))
    chunks = relevant_chunks(question, store.search(question, 6, sources=list(SOURCES)))
    raw_inputs = "\n".join(c.text for c in chunks) + result.answer
    assert any(seed.value in raw_inputs for seed in PII_SEEDS), "precondition: PII was in play"

    dumped = "\n".join(s.to_json() for s in exporter.get_finished_spans())
    dumped += json.dumps(
        [dict(s.resource.attributes) for s in exporter.get_finished_spans()], default=str
    )
    for seed in PII_SEEDS:
        assert seed.value not in dumped, seed.kind
    for secret in SECRETS:
        assert secret not in dumped


# --- real provider path (HTTP mocked, no network) -----------------------------------------


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_762_000_000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 321, "completion_tokens": 17, "total_tokens": 338},
    }


def test_real_openai_path_with_mocked_http(
    store: ChunkStore, exporter: InMemorySpanExporter
) -> None:
    # openai>=3 talks HTTP through `httpx2`, so the transport is injected directly (respx only
    # patches `httpx`). The no_network fixture would fail the test if anything leaked out.
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.url == "https://api.openai.com/v1/chat/completions"
        reply = "Priya Sharma owns the Q3 auth migration plan v2 [1][9]."
        return httpx2.Response(200, json=_completion(reply))

    transport = httpx2.MockTransport(handler)
    llm = make_chat_model(
        _settings(llm_provider="openai", llm_model="test-model", llm_api_key=SECRETS[0]),
        client_options={  # the graph calls ainvoke, which uses the async client
            "http_client": httpx2.Client(transport=transport),
            "http_async_client": httpx2.AsyncClient(transport=transport),
        },
    )
    result = ask(OWNER_Q, _deps(store, exporter, llm))

    [request] = requests
    assert request.headers["Authorization"] == f"Bearer {SECRETS[0]}"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert "never instructions" in body["messages"][0]["content"]
    assert '<evidence n="1"' in body["messages"][1]["content"]

    assert (result.token_usage.input_tokens, result.token_usage.output_tokens) == (321, 17)
    assert result.citation_report.unknown == [9]
    assert [c.n for c in result.citations] == [1]
    generation = next(s for s in exporter.get_finished_spans() if s.name == "llm.generate")
    assert generation.attributes is not None
    assert generation.attributes["gen_ai.usage.input_tokens"] == 321
    dumped = "\n".join(s.to_json() for s in exporter.get_finished_spans())
    assert SECRETS[0] not in dumped
