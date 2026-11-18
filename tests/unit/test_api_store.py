"""RunStore oldest-first eviction."""

from __future__ import annotations

import pytest

from sentinel.api.store import RunStore, StoredRun
from sentinel.graph.build import AskResult
from sentinel.graph.state import CitationReport, TokenUsage

pytestmark = pytest.mark.unit


def _result(question: str = "q") -> AskResult:
    return AskResult(
        question=question,
        answer="a",
        citations=[],
        insufficient_evidence=False,
        citation_report=CitationReport(),
        searched_sources=[],
        evidence_keys=[],
        plan=[],
        hops=0,
        unanswered=[],
        tool_calls=[],
        failures=[],
        stale_keys=[],
        errors=[],
        guard_notes=[],
        input_check=None,
        thread_id=None,
        turn=1,
        memory_notes=[],
        memory_writes=[],
        actions=[],
        output_redactions=[],
        session_id="s",
        trace_id="0" * 32,
        latency_ms=1,
        token_usage=TokenUsage(),
    )


def test_run_store_evicts_oldest_first() -> None:
    store = RunStore(max_runs=2)
    a = StoredRun(run_id="a", status="completed", result=_result("a"), events=[])
    b = StoredRun(run_id="b", status="completed", result=_result("b"), events=[])
    c = StoredRun(run_id="c", status="completed", result=_result("c"), events=[])
    store.put(a)
    store.put(b)
    store.put(c)
    assert len(store) == 2
    assert store.get("a") is None
    assert store.get("b") is not None
    assert store.get("c") is not None
