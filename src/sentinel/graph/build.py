"""Graph wiring.

START -> load_memory -> plan -> route -> execute_tools -> retrieve -> retrieved_guard -> verify
    verify -+-> route (next hop)
            +-> answer -> citation_check -> save_memory -> record_turn -> END

One span per node and per tool call. The plan is fixed by the planner from the user question;
the loop can only route sub-questions already in it (or widen an empty search to the other
allowed sources), bounded by `max_hops`. Tool results, index chunks and recalled memory are
scanned for injected instructions before routing, the verifier or the answerer can read them.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import anyio
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Tracer
from pydantic import BaseModel

from sentinel.citations import Citation
from sentinel.graph.conversation import context_from, prune_memory
from sentinel.graph.nodes.answerer import build_messages, insufficient_answer, is_insufficient
from sentinel.graph.nodes.citation_check import check_answer
from sentinel.graph.nodes.execute_tools import ToolExecutor, execute_calls
from sentinel.graph.nodes.planner import Planner, RulePlanner
from sentinel.graph.nodes.retrieve import assemble_evidence, relevant_chunks
from sentinel.graph.nodes.retrieved_guard import (
    guard_all,
    guard_memory,
    merge_notes,
    recheck_relevance,
)
from sentinel.graph.nodes.save_memory import (
    apply_writes,
    fact_from_answer,
    flag_conflicts,
    propose_writes,
)
from sentinel.graph.nodes.tool_router import (
    MEMORY_BUCKET,
    memory_refresh_calls,
    ready,
    route,
)
from sentinel.graph.nodes.verifier import decide as decide_next
from sentinel.graph.nodes.verifier import find_stale, superseded_by
from sentinel.graph.state import (
    AgentState,
    CitationReport,
    GuardNote,
    SubQuestion,
    TokenUsage,
    ToolCall,
    ToolFailure,
    TurnRecord,
    new_turn_input,
)
from sentinel.guards.injection import InjectionDetector
from sentinel.guards.sanitize import GuardMode
from sentinel.memory.backend import MemoryBackend
from sentinel.memory.freshness import find_conflicts
from sentinel.memory.models import MemoryItem, MemoryWrite
from sentinel.memory.policy import decide
from sentinel.retrieval.models import SOURCES, Source
from sentinel.retrieval.store import ChunkStore
from sentinel.tracing import generation_attributes, set_text, trace_id_hex

DEFAULT_K = 6
MEMORY_NOTES = 5


class Node(Protocol):
    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...


@dataclass
class Deps:
    store: ChunkStore
    llm: BaseChatModel
    tracer: Tracer
    tools: ToolExecutor | None = None
    """MCP tool access. None = index-only (no live tool calls)."""
    planner: Planner = field(default_factory=RulePlanner)
    k: int = DEFAULT_K
    max_hops: int = 4
    tool_timeout_s: float = 10.0
    tool_retries: int = 2
    backoff_s: float = 0.5
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep
    checkpointer: BaseCheckpointSaver[Any] | None = None
    """Short-term memory per thread_id. None = every question starts from scratch."""
    memory: MemoryBackend | None = None
    """Long-term memory. None = nothing is recalled or saved across threads."""
    data_mode: Literal["fixture", "live"] = "fixture"
    memory_auto_approve: bool = False
    """Write memory without approval; only honoured with fixture data (see memory.policy)."""
    injection_detector: InjectionDetector | None = field(default_factory=InjectionDetector)
    """Retrieved-content guard. None disables it (only for experiments and eval baselines)."""
    guard_mode: GuardMode = "remove"


def _model_name(llm: BaseChatModel) -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return llm._llm_type


def _guarded(notes: Sequence[GuardNote]) -> dict[str, int]:
    """Record key -> number of neutralized spans (summed over chunks and hops)."""
    counts: dict[str, int] = {}
    for note in notes:
        counts[note.key] = counts.get(note.key, 0) + note.spans
    return counts


def _usage(reply: object) -> TokenUsage:
    usage: Mapping[str, Any] = (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )


def build_graph(deps: Deps) -> Any:
    def traced(name: str, fn: Node) -> Node:
        async def run(state: AgentState) -> dict[str, Any]:
            with deps.tracer.start_as_current_span(f"node.{name}") as span:
                span.set_attribute("sentinel.hop", state.hops)
                return await fn(state)

        return run

    async def load_memory(state: AgentState) -> dict[str, Any]:
        if deps.memory is None:
            return {"memory_notes": []}
        notes = deps.memory.search(state.question, MEMORY_NOTES)
        flagged: list[GuardNote] = []
        if deps.injection_detector is not None:
            notes, flagged = guard_memory(notes, deps.injection_detector, deps.guard_mode)
        with deps.tracer.start_as_current_span("memory.load") as span:
            span.set_attribute("sentinel.memory.keys", [n.key for n in notes])
            span.set_attribute("sentinel.memory.verified", sum(n.verified for n in notes))
            span.set_attribute("sentinel.guard.flagged", [n.key for n in flagged])
        return {"memory_notes": notes, "guard_notes": merge_notes(state.guard_notes, flagged)}

    async def plan(state: AgentState) -> dict[str, Any]:
        context = context_from(state.turns, state.documents)
        result, notes, tokens = deps.planner.plan(state.question, state.sources, context)
        with deps.tracer.start_as_current_span("planner.plan") as span:
            set_text(span, "sentinel.plan", result.model_dump_json())
            span.set_attribute(
                "sentinel.follow_up", any(sq.prior_keys for sq in result.sub_questions)
            )
        return {
            "plan": result.sub_questions,
            "errors": [*state.errors, *notes],
            "token_usage": state.token_usage + tokens,
        }

    async def route_node(state: AgentState) -> dict[str, Any]:
        batch = ready(state.plan, state.done)
        hop = state.hops + 1
        calls: list[ToolCall] = []
        if deps.tools is not None:
            calls = route(
                batch,
                hop=hop,
                as_of=state.as_of,
                documents=state.documents,
                sq_docs=state.sq_docs,
                stale_keys=state.stale_keys,
                allowed_sources=state.sources,
                widened=state.widened,
            )
            if state.hops == 0:  # first hop: re-read the sources of verified memory notes
                calls += memory_refresh_calls(state.memory_notes, hop)
        sq_docs = dict(state.sq_docs)
        for sq in batch:
            if sq.relation == "recall":  # evidence straight from this thread's memory
                sq_docs[sq.id] = [k for k in sq.prior_keys if k in state.documents]
        return {
            "hops": hop,
            "current": [sq.id for sq in batch],
            "pending_calls": calls,
            "sq_docs": sq_docs,
        }

    async def execute(state: AgentState) -> dict[str, Any]:
        if deps.tools is None or not state.pending_calls:
            return {"pending_calls": []}

        def record(call: ToolCall, attempts: int, error: str | None) -> None:
            with deps.tracer.start_as_current_span("tool.call") as span:
                span.set_attribute("sentinel.tool", f"{call.source}.{call.tool}")
                set_text(span, "sentinel.tool.args", json.dumps(call.args, sort_keys=True))
                span.set_attribute("sentinel.tool.attempts", attempts)
                span.set_attribute("sentinel.tool.ok", error is None)

        result = await execute_calls(
            state.pending_calls,
            deps.tools,
            state.documents,
            timeout_s=deps.tool_timeout_s,
            retries=deps.tool_retries,
            backoff_s=deps.backoff_s,
            sleep=deps.sleep,
            on_call=record,
        )
        sq_docs = dict(state.sq_docs)
        for sq_id, keys in result.sq_docs.items():
            sq_docs[sq_id] = list(dict.fromkeys([*sq_docs.get(sq_id, []), *keys]))
        return {
            "pending_calls": [],
            "tool_calls": [*state.tool_calls, *state.pending_calls],
            "documents": {**state.documents, **result.documents},
            "sq_docs": sq_docs,
            "failures": [*state.failures, *result.failures],
        }

    async def retrieve(state: AgentState) -> dict[str, Any]:
        chunks = list(state.index_chunks)
        seen = {c.chunk_id for c in chunks}
        candidates = state.candidates
        by_id = {sq.id: sq for sq in state.plan}
        for sq_id in state.current:
            sq = by_id[sq_id]
            if sq.relation != "search" or sq_id in state.widened:
                continue
            hits = deps.store.search(sq.query or state.question, deps.k, sources=sq.target_sources)
            candidates += len(hits)
            for chunk in relevant_chunks(state.question, hits):
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    chunks.append(chunk)
        return {"index_chunks": chunks, "candidates": candidates}

    async def retrieved_guard(state: AgentState) -> dict[str, Any]:
        if deps.injection_detector is None:
            return {}
        documents, chunks, notes = guard_all(
            state.documents, state.index_chunks, deps.injection_detector, deps.guard_mode
        )
        all_notes = merge_notes(state.guard_notes, notes)
        queries: dict[str, list[str]] = {}
        for call in state.tool_calls:
            query = call.args.get("query")
            if call.hop == state.hops and isinstance(query, str) and query:
                queries.setdefault(call.sub_question_id, []).append(query)
        sq_docs, dropped = recheck_relevance(
            state.sq_docs, documents, queries, {n.key for n in all_notes}
        )
        with deps.tracer.start_as_current_span("guard.retrieved") as span:
            span.set_attribute("sentinel.guard.scanned", len(documents) + len(chunks))
            span.set_attribute("sentinel.guard.flagged", [n.key for n in notes])
            span.set_attribute("sentinel.guard.rules", sorted({r for n in notes for r in n.rules}))
            span.set_attribute("sentinel.guard.dropped", dropped)
        return {
            "documents": documents,
            "index_chunks": chunks,
            "sq_docs": sq_docs,
            "guard_notes": all_notes,
        }

    async def verify(state: AgentState) -> dict[str, Any]:
        done = list(dict.fromkeys([*state.done, *state.current]))
        stale = find_stale([*state.documents.values(), *state.index_chunks])
        index_keys = {c.doc_key for c in state.index_chunks}
        has_evidence = {
            sq.id: bool(state.sq_docs.get(sq.id)) or (sq.relation == "search" and bool(index_keys))
            for sq in state.plan
        }
        step, widen, unanswered = decide_next(
            state.plan,
            done,
            hops=state.hops,
            max_hops=deps.max_hops,
            has_evidence=has_evidence,
            widened=state.widened,
            allowed_sources=state.sources,
            tools_available=deps.tools is not None,
        )
        with deps.tracer.start_as_current_span("verifier.decide") as span:
            span.set_attribute("sentinel.next_step", step)
            span.set_attribute("sentinel.unanswered", unanswered)
            span.set_attribute("sentinel.stale", stale)
        return {
            "done": [d for d in done if d not in widen],
            "widened": [*state.widened, *widen],
            "stale_keys": stale,
            "next_step": step,
            "unanswered": unanswered,
            "current": [],
        }

    async def answer(state: AgentState) -> dict[str, Any]:
        # Freshness rule: a note that conflicts with a newer source record is stale from now
        # on in this turn, whether or not the persistent stale-marking is approved yet.
        successors = superseded_by([*state.documents.values(), *state.index_chunks])
        conflicts = find_conflicts(
            state.memory_notes, state.documents, list(successors), successors
        )
        notes = flag_conflicts(
            state.memory_notes, {c.key: (c.reason, c.superseded_by) for c in conflicts}
        )
        evidence = assemble_evidence(
            state.question,
            state.plan,
            state.documents,
            state.sq_docs,
            state.index_chunks,
            state.stale_keys,
            extra_buckets=[MEMORY_BUCKET],
            guarded=_guarded(state.guard_notes),
        )
        if not evidence:
            draft = insufficient_answer(
                state.question, state.sources, state.candidates, state.failures
            )
            return {"draft": draft, "evidence": [], "memory_notes": notes}
        messages = build_messages(
            state.question,
            evidence,
            state.plan,
            state.failures,
            state.unanswered,
            history=[t.question for t in state.turns][-3:],
            memory_notes=notes,
        )
        with deps.tracer.start_as_current_span("llm.generate") as span:
            set_text(span, "langfuse.observation.input", str(messages[-1].content))
            reply = await deps.llm.ainvoke(messages)
            tokens = _usage(reply)
            span.set_attributes(
                generation_attributes(
                    _model_name(deps.llm), tokens.input_tokens, tokens.output_tokens
                )
            )
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            set_text(span, "langfuse.observation.output", text)
        return {
            "draft": text,
            "evidence": evidence,
            "token_usage": state.token_usage + tokens,
            "memory_notes": notes,
        }

    async def citation_check(state: AgentState) -> dict[str, Any]:
        unverified = [n.value for n in state.memory_notes if not n.verified and not n.stale]
        final, used, report = check_answer(state.draft, state.evidence, unverified)
        return {
            "answer": final,
            "citations": used,
            "citation_report": report,
            "insufficient_evidence": is_insufficient(final),
        }

    async def save_memory(state: AgentState) -> dict[str, Any]:
        if deps.memory is None:
            return {"memory_writes": []}
        decision = decide(deps.data_mode, deps.memory_auto_approve)
        fact = fact_from_answer(
            state.question, state.answer, state.citations, state.citation_report, state.evidence
        )
        conflicting = [n for n in state.memory_notes if n.stale]
        existing = deps.memory.get(fact.key) if fact is not None else None
        writes = apply_writes(propose_writes(conflicting, fact, existing), deps.memory, decision)
        with deps.tracer.start_as_current_span("memory.save") as span:
            span.set_attribute("sentinel.memory.decision", decision)
            span.set_attribute("sentinel.memory.writes", [f"{w.op}:{w.status}" for w in writes])
        return {"memory_writes": writes}

    async def record_turn(state: AgentState) -> dict[str, Any]:
        turn = TurnRecord(
            question=state.question,
            answer=state.answer,
            as_of=state.as_of,
            plan=state.plan,
            unanswered=state.unanswered,
            sq_docs=state.sq_docs,
            evidence_keys=[ev.citation.key for ev in state.evidence],
            cited_keys=[c.key for c in state.citations],
            failed_sources=sorted({f.source for f in state.failures}),
        )
        turns, documents = prune_memory([*state.turns, turn], state.documents)
        return {"turns": turns, "documents": documents}

    graph = StateGraph(AgentState)
    for name, fn in (
        ("load_memory", load_memory),
        ("plan", plan),
        ("route", route_node),
        ("execute_tools", execute),
        ("retrieve", retrieve),
        ("retrieved_guard", retrieved_guard),
        ("verify", verify),
        ("answer", answer),
        ("citation_check", citation_check),
        ("save_memory", save_memory),
        ("record_turn", record_turn),
    ):
        graph.add_node(name, traced(name, fn))
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "plan")
    graph.add_edge("plan", "route")
    graph.add_edge("route", "execute_tools")
    graph.add_edge("execute_tools", "retrieve")
    graph.add_edge("retrieve", "retrieved_guard")
    graph.add_edge("retrieved_guard", "verify")
    graph.add_conditional_edges(
        "verify", lambda s: s.next_step, {"route": "route", "answer": "answer"}
    )
    graph.add_edge("answer", "citation_check")
    graph.add_edge("citation_check", "save_memory")
    graph.add_edge("save_memory", "record_turn")
    graph.add_edge("record_turn", END)
    return graph.compile(checkpointer=deps.checkpointer)


class AskResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    citation_report: CitationReport
    searched_sources: list[Source]
    evidence_keys: list[str]
    plan: list[SubQuestion]
    hops: int
    unanswered: list[str]
    tool_calls: list[ToolCall]
    failures: list[ToolFailure]
    stale_keys: list[str]
    errors: list[str]
    guard_notes: list[GuardNote]
    """Retrieved records in which suspected injected instructions were neutralized."""
    thread_id: str | None
    """Set when a checkpointer is configured; pass it back to continue the conversation."""
    turn: int
    """1 for a new thread (or no checkpointer), n for the n-th question of a thread."""
    memory_notes: list[MemoryItem]
    """Long-term memory consulted this turn (with this turn's conflicts flagged stale)."""
    memory_writes: list[MemoryWrite]
    trace_id: str
    latency_ms: int
    token_usage: TokenUsage


async def ask_async(
    question: str,
    deps: Deps,
    sources: Sequence[Source] | None = None,
    as_of: datetime | None = None,
    thread_id: str | None = None,
) -> AskResult:
    """Answer one question. With `deps.checkpointer`, the turn joins `thread_id`'s short-term
    memory (a new id is generated when None); without one, nothing is remembered."""
    graph = build_graph(deps)
    chosen = list(sources or SOURCES)
    if deps.checkpointer is not None and thread_id is None:
        thread_id = uuid.uuid4().hex
    if deps.checkpointer is None:
        thread_id = None
    config: dict[str, Any] = {"recursion_limit": 10 + 6 * deps.max_hops}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    turn_input = new_turn_input(question, chosen, as_of or datetime.now(UTC))
    with deps.tracer.start_as_current_span("sentinel.ask") as root:
        trace_id = trace_id_hex(root)
        root.set_attribute("langfuse.trace.name", "sentinel.ask")
        if thread_id is not None:
            root.set_attribute("langfuse.session.id", thread_id)  # groups a thread's traces
        set_text(root, "langfuse.trace.input", question)
        started = time.perf_counter()
        out = await graph.ainvoke(turn_input, config=config)
        state = AgentState.model_validate(out)
        latency_ms = int((time.perf_counter() - started) * 1000)
        set_text(root, "langfuse.trace.output", state.answer)
        root.set_attribute("sentinel.hops", state.hops)
        root.set_attribute("sentinel.tool_calls", len(state.tool_calls))
        root.set_attribute("sentinel.tool_failures", len(state.failures))
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
        plan=state.plan,
        hops=state.hops,
        unanswered=state.unanswered,
        tool_calls=state.tool_calls,
        failures=state.failures,
        stale_keys=state.stale_keys,
        errors=state.errors,
        guard_notes=state.guard_notes,
        thread_id=thread_id,
        turn=len(state.turns),
        memory_notes=state.memory_notes,
        memory_writes=state.memory_writes,
        trace_id=trace_id,
        latency_ms=latency_ms,
        token_usage=state.token_usage,
    )


def ask(
    question: str,
    deps: Deps,
    sources: Sequence[Source] | None = None,
    as_of: datetime | None = None,
) -> AskResult:
    """Synchronous wrapper; use `ask_async` inside an event loop."""

    async def run() -> AskResult:
        return await ask_async(question, deps, sources, as_of)

    return anyio.run(run)


async def thread_turns(deps: Deps, thread_id: str) -> list[TurnRecord]:
    """The remembered turns of a thread (empty for unknown ids or without a checkpointer)."""
    if deps.checkpointer is None:
        return []
    snapshot = await build_graph(deps).aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    return [TurnRecord.model_validate(t) for t in values.get("turns", [])]
