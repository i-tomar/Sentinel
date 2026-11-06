"""Graph wiring: START -> retrieve -> answer -> citation_check -> END, one span per node."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from sentinel.citations import Citation
from sentinel.graph.nodes.answerer import build_messages, insufficient_answer, is_insufficient
from sentinel.graph.nodes.citation_check import check_answer
from sentinel.graph.nodes.retrieve import build_evidence
from sentinel.graph.state import AgentState, CitationReport, TokenUsage
from sentinel.retrieval.models import SOURCES, Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import generation_attributes, set_text, trace_id_hex

DEFAULT_K = 6


class Node(Protocol):
    def __call__(self, state: AgentState) -> dict[str, Any]: ...


@dataclass
class Deps:
    store: ChunkStore
    llm: BaseChatModel
    tracer: Tracer
    k: int = DEFAULT_K


def _model_name(llm: BaseChatModel) -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return llm._llm_type


def build_graph(deps: Deps) -> Any:
    def traced(name: str, fn: Node) -> Node:
        def run(state: AgentState) -> dict[str, Any]:
            with deps.tracer.start_as_current_span(f"node.{name}"):
                return fn(state)

        return run

    def retrieve(state: AgentState) -> dict[str, Any]:
        hits = deps.store.search(state.question, deps.k, sources=state.sources)
        evidence = build_evidence(state.question, hits)
        with deps.tracer.start_as_current_span("retrieve.hits") as span:
            span.set_attribute("sentinel.candidates", len(hits))
            span.set_attribute("sentinel.evidence", [ev.citation.key for ev in evidence])
        return {"candidates": len(hits), "evidence": evidence}

    def answer(state: AgentState) -> dict[str, Any]:
        if not state.evidence:
            draft = insufficient_answer(state.question, state.sources, state.candidates)
            return {"draft": draft}
        messages = build_messages(state.question, state.evidence)
        with deps.tracer.start_as_current_span("llm.generate") as span:
            set_text(span, "langfuse.observation.input", str(messages[-1].content))
            reply = deps.llm.invoke(messages)
            usage: Mapping[str, Any] = (
                (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
            )
            tokens = TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            )
            span.set_attributes(
                generation_attributes(
                    _model_name(deps.llm), tokens.input_tokens, tokens.output_tokens
                )
            )
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            set_text(span, "langfuse.observation.output", text)
        return {"draft": text, "token_usage": tokens}

    def citation_check(state: AgentState) -> dict[str, Any]:
        final, used, report = check_answer(state.draft, state.evidence)
        return {
            "answer": final,
            "citations": used,
            "citation_report": report,
            "insufficient_evidence": is_insufficient(final),
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", traced("retrieve", retrieve))
    graph.add_node("answer", traced("answer", answer))
    graph.add_node("citation_check", traced("citation_check", citation_check))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "citation_check")
    graph.add_edge("citation_check", END)
    return graph.compile()


class AskResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    citation_report: CitationReport
    searched_sources: list[Source]
    evidence_keys: list[str]
    trace_id: str
    latency_ms: int
    token_usage: TokenUsage


def ask(question: str, deps: Deps, sources: Sequence[Source] | None = None) -> AskResult:
    graph = build_graph(deps)
    chosen = list(sources or SOURCES)
    with deps.tracer.start_as_current_span("sentinel.ask") as root:
        trace_id = trace_id_hex(root)
        root.set_attribute("langfuse.trace.name", "sentinel.ask")
        set_text(root, "langfuse.trace.input", question)
        started = time.perf_counter()
        out = graph.invoke(AgentState(question=question, sources=chosen))
        state = AgentState.model_validate(out)
        latency_ms = int((time.perf_counter() - started) * 1000)
        set_text(root, "langfuse.trace.output", state.answer)
        root.set_attribute("sentinel.insufficient_evidence", state.insufficient_evidence)
        root.set_attribute("sentinel.citation_check.ok", state.citation_report.ok)
        root.set_attribute("sentinel.citations.unknown", len(state.citation_report.unknown))
    return AskResult(
        question=question,
        answer=state.answer,
        citations=state.citations,
        insufficient_evidence=state.insufficient_evidence,
        citation_report=state.citation_report,
        searched_sources=chosen,
        evidence_keys=[ev.citation.key for ev in state.evidence],
        trace_id=trace_id,
        latency_ms=latency_ms,
        token_usage=state.token_usage,
    )
