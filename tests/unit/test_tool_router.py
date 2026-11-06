from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sentinel.graph.nodes.tool_router import MAX_ENTITIES, route, ticket_keys, window_dates
from sentinel.graph.state import SubQuestion
from sentinel.mcp_clients.tools import READ_TOOLS
from sentinel.retrieval.models import Document, Source

pytestmark = pytest.mark.unit

MONDAY = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def _doc(source: Source, doc_id: str, text: str) -> Document:
    return Document(
        source=source,
        id=doc_id,
        url=f"https://x.test/{doc_id}",
        title=doc_id,
        text=text,
        updated_at=MONDAY,
        fetched_at=MONDAY,
    )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ("last_week", (date(2026, 10, 26), date(2026, 11, 2))),
        ("this_week", (date(2026, 11, 2), date(2026, 11, 3))),
        ("last_7_days", (date(2026, 10, 26), date(2026, 11, 3))),
    ],
)
def test_window_dates(window: str, expected: tuple[date, date]) -> None:
    assert window_dates(window, MONDAY) == expected  # type: ignore[arg-type]
    assert window_dates(window, MONDAY.replace(day=6))[0] <= date(2026, 11, 6)  # type: ignore[arg-type]


def test_ticket_keys_are_strict_and_ordered() -> None:
    docs = [
        _doc("notion", "plan", "Critical path PROJ-120; blocks PROJ-142 and PROJ-147. PROJ-120."),
        _doc("jira", "PROJ-151", "see proj-999 (lowercase ignored) and update_issue(PROJ-1)"),
    ]
    assert ticket_keys(docs) == ["PROJ-120", "PROJ-142", "PROJ-147", "PROJ-151", "PROJ-1"]


SEARCH = SubQuestion(
    id="sq1",
    text="Find the plan",
    target_sources=["notion", "gmail"],
    query="auth migration",
    time_window="last_week",
    label="escalation",
)
BLOCKED = SubQuestion(
    id="sq2", text="b", target_sources=["jira"], depends_on=["sq1"], relation="blocked_by"
)
MENTIONS = SubQuestion(
    id="sq3",
    text="m",
    target_sources=["gmail"],
    depends_on=["sq2"],
    relation="mentions",
    time_window="last_week",
)


def test_search_calls_carry_gmail_filters_only_for_gmail() -> None:
    calls = route(
        [SEARCH],
        hop=1,
        as_of=MONDAY,
        documents={},
        sq_docs={},
        stale_keys=[],
        allowed_sources=["notion", "jira", "gmail"],
    )
    assert [(c.source, c.tool) for c in calls] == [
        ("notion", "search_pages"),
        ("gmail", "search_threads"),
    ]
    assert calls[0].args == {"query": "auth migration", "limit": 5}
    assert calls[1].args == {
        "query": "auth migration",
        "limit": 5,
        "after": "2026-10-26",
        "before": "2026-11-02",
        "label": "escalation",
    }
    assert [c.id for c in calls] == ["h1-c1", "h1-c2"]


def test_blocked_by_uses_only_fresh_dependency_keys_and_caps_them() -> None:
    current = _doc("notion", "plan-v2", " ".join(f"PROJ-{n}" for n in range(100, 110)))
    stale = _doc("notion", "plan-v1", "PROJ-133")
    calls = route(
        [BLOCKED],
        hop=2,
        as_of=MONDAY,
        documents={current.key: current, stale.key: stale},
        sq_docs={"sq1": [stale.key, current.key]},
        stale_keys=[stale.key],
        allowed_sources=["notion", "jira", "gmail"],
    )
    assert len(calls) == MAX_ENTITIES
    assert {c.tool for c in calls} == {"search_issues"}
    assert [c.args["blocked_by"] for c in calls][:2] == ["PROJ-100", "PROJ-101"]
    assert "PROJ-133" not in {c.args["blocked_by"] for c in calls}


def test_mentions_only_uses_jira_records() -> None:
    jira = _doc("jira", "PROJ-142", "blocked")
    notion = _doc("notion", "plan", "PROJ-999")
    calls = route(
        [MENTIONS],
        hop=3,
        as_of=MONDAY,
        documents={jira.key: jira, notion.key: notion},
        sq_docs={"sq2": [jira.key, notion.key]},
        stale_keys=[],
        allowed_sources=["notion", "jira", "gmail"],
    )
    assert [c.args["query"] for c in calls] == ["PROJ-142"]
    assert calls[0].args["after"] == "2026-10-26"


def test_widened_search_tries_only_the_other_sources() -> None:
    sq = SEARCH.model_copy(update={"target_sources": ["notion"], "time_window": None})
    calls = route(
        [sq],
        hop=2,
        as_of=MONDAY,
        documents={},
        sq_docs={},
        stale_keys=[],
        allowed_sources=["notion", "jira", "gmail"],
        widened=["sq1"],
    )
    assert [c.source for c in calls] == ["jira", "gmail"]


def test_every_routed_call_is_a_read_tool() -> None:
    jira = _doc("jira", "PROJ-142", "x")
    for sq in (SEARCH, BLOCKED, MENTIONS):
        calls = route(
            [sq],
            hop=1,
            as_of=MONDAY,
            documents={jira.key: jira},
            sq_docs={"sq1": [jira.key], "sq2": [jira.key]},
            stale_keys=[],
            allowed_sources=["notion", "jira", "gmail"],
        )
        assert all((c.source, c.tool) in READ_TOOLS for c in calls)
