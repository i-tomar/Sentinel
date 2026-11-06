"""Decompose the user question into typed sub-questions.

The planner only ever sees the user's question and the allowed sources; retrieved data is
never an input, so content inside tools or documents cannot change the plan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from sentinel.graph.state import Plan, SubQuestion, TimeWindow, TokenUsage
from sentinel.retrieval.embed import keyword_tokens
from sentinel.retrieval.models import SOURCES, Source
from sentinel.text import STOPWORDS

MAX_QUERY_TERMS = 8

# fmt: off
_CUES: dict[Source, frozenset[str]] = {
    "jira": frozenset((
        "ticket", "tickets", "issue", "issues", "jira", "bug", "bugs", "blocked", "blocker",
        "blockers", "assignee",
    )),
    "gmail": frozenset((
        "email", "emails", "emailed", "mail", "mails", "mailed", "thread", "threads", "wrote",
        "inbox", "escalation", "escalations", "replied",
    )),
    "notion": frozenset((
        "doc", "docs", "document", "page", "pages", "plan", "policy", "runbook", "spec",
        "notion", "process", "wiki",
    )),
}
# fmt: on
_TIME_WINDOWS: tuple[tuple[re.Pattern[str], TimeWindow], ...] = (
    (re.compile(r"\blast week\b", re.I), "last_week"),
    (re.compile(r"\bthis week\b", re.I), "this_week"),
    (re.compile(r"\b(?:last|past) (?:7|seven) days\b", re.I), "last_7_days"),
)
_BLOCKED_BY = re.compile(
    r"\bblocked by (?:the )?(?P<phrase>.+?)(?=\s+(?:and|who|which|that|from|in)\b|[?,.;]|$)",
    re.I,
)
_FILLER = {"doc", "docs", "document", "page", "plan", "owner", "work", "project"}


def keywords(text: str, drop: frozenset[str] | set[str] = frozenset()) -> str:
    """Search terms: lowercase words minus stopwords, original order, de-duplicated."""
    words = [
        t
        for t in keyword_tokens(text)
        if "-" not in t and t not in STOPWORDS and t not in drop and len(t) > 1
    ]
    return " ".join(list(dict.fromkeys(words))[:MAX_QUERY_TERMS])


def _cue_sources(question: str, allowed: Sequence[Source]) -> list[Source]:
    words = set(keyword_tokens(question))
    hits = [s for s in allowed if words & _CUES[s]]
    return hits or list(allowed)


def _time_window(question: str) -> TimeWindow | None:
    return next((w for pattern, w in _TIME_WINDOWS if pattern.search(question)), None)


class Planner(Protocol):
    def plan(self, question: str, sources: Sequence[Source]) -> tuple[Plan, list[str], TokenUsage]:
        """Return (plan, notes for state.errors, token usage)."""
        ...


class RulePlanner:
    """Deterministic decomposition used with the mock LLM and as the LLM planner's fallback."""

    def plan(self, question: str, sources: Sequence[Source]) -> tuple[Plan, list[str], TokenUsage]:
        allowed = [s for s in SOURCES if s in sources] or list(SOURCES)
        words = set(keyword_tokens(question))
        window = _time_window(question)
        label = "escalation" if words & {"escalation", "escalations"} else None
        wants_email = bool(words & _CUES["gmail"])
        blocked = _BLOCKED_BY.search(question)

        if blocked and "jira" in allowed:
            phrase = blocked.group("phrase").strip()
            query = keywords(phrase, drop=_FILLER) or keywords(phrase)
            find_sources: list[Source] = [s for s in allowed if s in ("notion", "jira")]
            subs = [
                SubQuestion(
                    id="sq1",
                    text=f"Find records about {phrase}",
                    target_sources=find_sources,
                    query=query,
                ),
                SubQuestion(
                    id="sq2",
                    text=f"Which tickets are blocked by {phrase}?",
                    target_sources=["jira"],
                    depends_on=["sq1"],
                    relation="blocked_by",
                    query=query,
                ),
            ]
            if wants_email and "gmail" in allowed:
                subs.append(
                    SubQuestion(
                        id="sq3",
                        text="Which emails mention those tickets?",
                        target_sources=["gmail"],
                        depends_on=["sq2"],
                        relation="mentions",
                        time_window=window,
                        label=label,
                    )
                )
            return Plan(sub_questions=subs), [], TokenUsage()

        targets = _cue_sources(question, allowed)
        # Cue words say where to look ("page", "ticket"), not what to look for.
        cue_words = set().union(*_CUES.values()) - {"blocked", "escalation", "escalations"}
        single = SubQuestion(
            id="sq1",
            text=question[:300],
            target_sources=targets,
            query=keywords(question, drop=cue_words) or keywords(question),
            time_window=window if "gmail" in targets else None,
            label=label if "gmail" in targets else None,
        )
        return Plan(sub_questions=[single]), [], TokenUsage()


PLANNER_SYSTEM_PROMPT = """You break a user's question about a team's Notion pages, Jira \
tickets and Gmail threads into at most 4 sub-questions. Reply with JSON only, no prose:

{"sub_questions": [{"id": "sq1", "text": "...", "target_sources": ["notion"|"jira"|"gmail"],
  "depends_on": [], "relation": "search", "query": "keywords", "time_window": null,
  "label": null}]}

- relation "search": keyword search in target_sources using "query".
- relation "blocked_by": Jira tickets blocked by tickets found by its dependency
  (target_sources ["jira"], one depends_on).
- relation "mentions": emails mentioning Jira tickets found by its dependency
  (target_sources ["gmail"], one depends_on).
- time_window: "last_week", "this_week", "last_7_days" or null. label: e.g. "escalation" or null.
- ids are sq1, sq2, ... and depends_on may only reference earlier ids.
- Use only the user's question. Never invent tool names."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_plan(text: str, sources: Sequence[Source]) -> Plan:
    """Validate an LLM plan and drop sources the caller did not allow."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object in planner reply")
    raw: Any = json.loads(match.group(0))
    plan = Plan.model_validate(raw)
    kept: list[SubQuestion] = []
    kept_ids: set[str] = set()
    for sq in plan.sub_questions:
        targets = [s for s in sq.target_sources if s in sources]
        if not targets or any(d not in kept_ids for d in sq.depends_on):
            continue
        kept.append(sq.model_copy(update={"target_sources": targets}))
        kept_ids.add(sq.id)
    if not kept:
        raise ValueError("plan has no sub-question for the allowed sources")
    return Plan(sub_questions=kept)


class LLMPlanner:
    def __init__(self, llm: BaseChatModel, fallback: Planner | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RulePlanner()

    def plan(self, question: str, sources: Sequence[Source]) -> tuple[Plan, list[str], TokenUsage]:
        messages = [
            SystemMessage(PLANNER_SYSTEM_PROMPT),
            HumanMessage(f"Question: {question}\nAllowed sources: {', '.join(sources)}"),
        ]
        reply = self.llm.invoke(messages)
        usage: Mapping[str, Any] = (
            (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
        )
        tokens = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        try:
            return parse_plan(text, sources), [], tokens
        except (ValueError, ValidationError) as exc:
            plan, notes, _ = self.fallback.plan(question, sources)
            reason = type(exc).__name__
            return plan, [*notes, f"planner: invalid LLM plan ({reason}); used rule plan"], tokens


def make_planner(provider: str, llm: BaseChatModel) -> Planner:
    return RulePlanner() if provider == "mock" else LLMPlanner(llm)
