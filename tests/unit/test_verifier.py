from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.graph.nodes.verifier import decide, find_stale, topic_key
from sentinel.graph.state import SubQuestion
from sentinel.retrieval.models import Document, Source

pytestmark = pytest.mark.unit


def _doc(source: Source, doc_id: str, title: str, day: int) -> Document:
    ts = datetime(2026, 10, day, tzinfo=UTC)
    return Document(
        source=source,
        id=doc_id,
        url="https://x.test",
        title=title,
        text="t",
        updated_at=ts,
        fetched_at=ts,
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Q3 Auth Migration Plan (v1)", "Q3 Auth Migration Plan (v2)"),
        ("Pricing 2025", "Pricing 2026"),
        ("On-call Runbook 2025", "On-call Runbook 2026"),
    ],
)
def test_topic_key_ignores_versions_and_years(a: str, b: str) -> None:
    assert topic_key(a) == topic_key(b)


def test_find_stale_keeps_newest_per_topic_and_ignores_other_sources() -> None:
    docs = [
        _doc("notion", "pricing-2025", "Pricing 2025", 1),
        _doc("notion", "pricing-2026", "Pricing 2026", 20),
        _doc("notion", "team-directory", "Team Directory", 5),
        _doc("jira", "PROJ-1", "Pricing 2025", 1),
        _doc("jira", "PROJ-2", "Pricing 2026", 2),
    ]
    assert find_stale(docs) == ["notion:pricing-2025"]


PLAN = [
    SubQuestion(id="sq1", text="a", target_sources=["notion"], query="x"),
    SubQuestion(
        id="sq2", text="b", target_sources=["jira"], depends_on=["sq1"], relation="blocked_by"
    ),
]
ALL: list[Source] = ["notion", "jira", "gmail"]


def test_routes_while_sub_questions_are_ready() -> None:
    step, widen, unanswered = decide(
        PLAN,
        ["sq1"],
        hops=1,
        max_hops=4,
        has_evidence={"sq1": True},
        widened=[],
        allowed_sources=ALL,
        tools_available=True,
    )
    assert (step, widen, unanswered) == ("route", [], ["sq2"])


def test_answers_when_done() -> None:
    step, _, unanswered = decide(
        PLAN,
        ["sq1", "sq2"],
        hops=2,
        max_hops=4,
        has_evidence={"sq1": True, "sq2": True},
        widened=[],
        allowed_sources=ALL,
        tools_available=True,
    )
    assert (step, unanswered) == ("answer", [])


def test_max_hops_stops_the_loop() -> None:
    step, _, unanswered = decide(
        PLAN,
        ["sq1"],
        hops=1,
        max_hops=1,
        has_evidence={"sq1": True},
        widened=[],
        allowed_sources=ALL,
        tools_available=True,
    )
    assert (step, unanswered) == ("answer", ["sq2"])


def test_empty_search_is_widened_once() -> None:
    plan = PLAN[:1]
    args = {
        "hops": 1,
        "max_hops": 4,
        "has_evidence": {"sq1": False},
        "allowed_sources": ALL,
        "tools_available": True,
    }
    assert decide(plan, ["sq1"], widened=[], **args)[:2] == ("route", ["sq1"])  # type: ignore[arg-type]
    assert decide(plan, ["sq1"], widened=["sq1"], **args)[0] == "answer"  # type: ignore[arg-type]
    no_tools = {**args, "tools_available": False}
    assert decide(plan, ["sq1"], widened=[], **no_tools)[0] == "answer"  # type: ignore[arg-type]
    all_targeted = [plan[0].model_copy(update={"target_sources": ALL})]
    assert decide(all_targeted, ["sq1"], widened=[], **args)[0] == "answer"  # type: ignore[arg-type]
