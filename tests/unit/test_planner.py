from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from sentinel.graph.nodes.planner import (
    PLANNER_SYSTEM_PROMPT,
    LLMPlanner,
    RulePlanner,
    keywords,
    parse_plan,
)
from sentinel.llm import MockLLM, prompt_hash
from sentinel.retrieval.models import SOURCES

pytestmark = pytest.mark.unit

THREE_HOP = "Which tickets are blocked by the auth migration and who emailed about them?"
ESCALATIONS = (
    "Which customer escalation emails from last week relate to tickets blocked by the "
    "Q3 auth migration?"
)


def test_keywords_drop_stopwords_and_keep_order() -> None:
    assert keywords("Who owns the Q3 auth migration plan v2?") == "owns q3 auth migration plan v2"


def test_three_hop_decomposition() -> None:
    plan, notes, _ = RulePlanner().plan(THREE_HOP, SOURCES)
    subs = plan.sub_questions
    assert notes == []
    assert [(s.id, s.relation, s.target_sources, s.depends_on) for s in subs] == [
        ("sq1", "search", ["notion", "jira"], []),
        ("sq2", "blocked_by", ["jira"], ["sq1"]),
        ("sq3", "mentions", ["gmail"], ["sq2"]),
    ]
    assert subs[0].query == "auth migration"


def test_time_window_and_label() -> None:
    plan, _, _ = RulePlanner().plan(ESCALATIONS, SOURCES)
    sq3 = plan.sub_questions[2]
    assert (sq3.time_window, sq3.label) == ("last_week", "escalation")
    assert plan.sub_questions[0].query == "q3 auth migration"


def test_simple_question_targets_cued_sources() -> None:
    plan, _, _ = RulePlanner().plan("Who replied in the Globex renewal email thread?", SOURCES)
    [sq] = plan.sub_questions
    assert sq.target_sources == ["gmail"]
    assert sq.relation == "search"
    no_cue, _, _ = RulePlanner().plan("Who is Priya Sharma?", SOURCES)
    assert no_cue.sub_questions[0].target_sources == ["notion", "jira", "gmail"]


def test_routing_words_are_not_search_terms() -> None:
    plan, _, _ = RulePlanner().plan("Which page explains the multipart export failure?", SOURCES)
    assert plan.sub_questions[0].target_sources == ["notion"]
    assert plan.sub_questions[0].query == "explains multipart export failure"


def test_plan_respects_allowed_sources() -> None:
    plan, _, _ = RulePlanner().plan(THREE_HOP, ["notion", "jira"])
    assert [s.id for s in plan.sub_questions] == ["sq1", "sq2"]  # no gmail hop
    only_gmail, _, _ = RulePlanner().plan(THREE_HOP, ["gmail"])
    assert [s.target_sources for s in only_gmail.sub_questions] == [["gmail"]]


# --- LLM planner --------------------------------------------------------------------------


def _llm_messages(question: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(PLANNER_SYSTEM_PROMPT),
        HumanMessage(f"Question: {question}\nAllowed sources: notion, jira, gmail"),
    ]


VALID_PLAN = {
    "sub_questions": [
        {"id": "sq1", "text": "Find the plan", "target_sources": ["notion"], "query": "auth"},
        {
            "id": "sq2",
            "text": "Blocked tickets",
            "target_sources": ["jira"],
            "depends_on": ["sq1"],
            "relation": "blocked_by",
        },
    ]
}


def test_llm_plan_is_parsed_from_fenced_json() -> None:
    reply = f"```json\n{json.dumps(VALID_PLAN)}\n```"
    llm = MockLLM(scripted={prompt_hash(_llm_messages(THREE_HOP)): reply})
    plan, notes, _ = LLMPlanner(llm).plan(THREE_HOP, SOURCES)
    assert notes == []
    assert [s.text for s in plan.sub_questions] == ["Find the plan", "Blocked tickets"]


@pytest.mark.parametrize(
    "reply",
    [
        "I think you should search everywhere.",
        json.dumps({"sub_questions": [{"id": "sq1", "text": "x", "target_sources": ["slack"]}]}),
        json.dumps(
            {
                "sub_questions": [
                    {"id": "sq1", "text": "x", "target_sources": ["jira"], "depends_on": ["sq2"]}
                ]
            }
        ),
        json.dumps(
            {
                "sub_questions": [
                    {"id": "sq1", "text": "x", "target_sources": ["jira"], "tool": "update_issue"}
                ]
            }
        ),
        json.dumps(
            {
                "sub_questions": [
                    {"id": "sq1", "text": "x", "target_sources": ["jira"], "relation": "blocked_by"}
                ]
            }
        ),
    ],
    ids=["prose", "unknown-source", "forward-dependency", "tool-field", "relation-no-dependency"],
)
def test_invalid_llm_plan_falls_back_to_rules(reply: str) -> None:
    llm = MockLLM(scripted={prompt_hash(_llm_messages(THREE_HOP)): reply})
    plan, notes, _ = LLMPlanner(llm).plan(THREE_HOP, SOURCES)
    assert notes
    assert notes[0].startswith("planner: invalid LLM plan")
    assert plan == RulePlanner().plan(THREE_HOP, SOURCES)[0]


def test_parse_plan_drops_disallowed_sources_and_orphans() -> None:
    raw = json.dumps(
        {
            "sub_questions": [
                {"id": "sq1", "text": "a", "target_sources": ["gmail"]},
                {"id": "sq2", "text": "b", "target_sources": ["jira"], "depends_on": ["sq1"]},
                {"id": "sq3", "text": "c", "target_sources": ["notion", "jira"]},
            ]
        }
    )
    plan = parse_plan(raw, ["notion", "jira"])
    assert [(s.id, s.target_sources) for s in plan.sub_questions] == [("sq3", ["notion", "jira"])]
