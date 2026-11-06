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

from sentinel.graph.conversation import ConversationContext
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


_ANAPHORA = re.compile(
    r"\b(?:them|those|these|they|it|its|that one|this one|same|above|previous|earlier"
    r"|that (?:ticket|issue|email|thread|page|doc)"
    r"|(?:first|second|third|fourth|fifth|last) (?:one|ticket|issue|email|thread|page|doc))\b",
    re.I,
)
_LEADING_FOLLOW_UP = re.compile(r"^\s*(?:and|also|what about|how about)\b", re.I)
_CONTINUE = re.compile(
    r"\b(?:continue|try again|retry|keep going|go on|finish (?:it|that|up)|the rest)\b", re.I
)
_EMAILS_ABOUT = re.compile(
    r"\b(?:emailed|e-?mails?|wrote|threads?|mails?)\s+(?:about|on|regarding|mentioning)\b"
    r"|\bwho (?:emailed|wrote|mailed)\b",
    re.I,
)
_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "last": -1}
# "the second ticket", not "last week"
_ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|last)\s+(?:one|ticket|issue|email|thread|page|doc)s?\b",
    re.I,
)
MAX_PRIOR_KEYS = 6


def is_follow_up(question: str) -> bool:
    return bool(
        _ANAPHORA.search(question)
        or _LEADING_FOLLOW_UP.match(question)
        or _CONTINUE.search(question)
    )


def _pick(question: str, keys: Sequence[str]) -> list[str]:
    """Apply an ordinal ("the second ticket") to remembered keys; otherwise keep them all."""
    match = _ORDINAL.search(question)
    if not match:
        return list(keys)[:MAX_PRIOR_KEYS]
    index = _ORDINALS[match.group(1).lower()]
    return [keys[index]] if -len(keys) <= index < len(keys) else []


class Planner(Protocol):
    def plan(
        self,
        question: str,
        sources: Sequence[Source],
        context: ConversationContext | None = None,
    ) -> tuple[Plan, list[str], TokenUsage]:
        """Return (plan, notes for state.errors, token usage)."""
        ...


class RulePlanner:
    """Deterministic decomposition used with the mock LLM and as the LLM planner's fallback."""

    def plan(
        self,
        question: str,
        sources: Sequence[Source],
        context: ConversationContext | None = None,
    ) -> tuple[Plan, list[str], TokenUsage]:
        allowed = [s for s in SOURCES if s in sources] or list(SOURCES)
        words = set(keyword_tokens(question))
        window = _time_window(question)
        label = "escalation" if words & {"escalation", "escalations"} else None
        wants_email = bool(words & _CUES["gmail"])
        blocked = _BLOCKED_BY.search(question)

        if context is not None and is_follow_up(question):
            follow_up = self._follow_up(question, allowed, context, window, label)
            if follow_up is not None:
                return follow_up, [], TokenUsage()

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

    def _follow_up(
        self,
        question: str,
        allowed: Sequence[Source],
        ctx: ConversationContext,
        window: TimeWindow | None,
        label: str | None,
    ) -> Plan | None:
        """Plan that builds on the thread's memory, or None to plan from scratch."""
        if _CONTINUE.search(question) and ctx.unresolved:
            return self._carry_over(allowed, ctx)
        referents = ctx.referents
        if not referents:
            return self._carry_over(allowed, ctx) if ctx.unresolved else None
        words = set(keyword_tokens(question))

        if _EMAILS_ABOUT.search(question) and "gmail" in allowed:
            tickets = _pick(question, [k for k in referents if k.startswith("jira:")])
            if tickets:
                ids = ", ".join(k.split(":", 1)[1] for k in tickets)
                return Plan(
                    sub_questions=[
                        SubQuestion(
                            id="sq1",
                            text=f"Which emails mention {ids}?",
                            target_sources=["gmail"],
                            relation="mentions",
                            prior_keys=tickets,
                            time_window=window,
                            label=label,
                        )
                    ]
                )
        if "blocked" in words and "jira" in allowed:
            keys = _pick(question, referents)
            if keys:
                return Plan(
                    sub_questions=[
                        SubQuestion(
                            id="sq1",
                            text="Which tickets are blocked by those records?",
                            target_sources=["jira"],
                            relation="blocked_by",
                            prior_keys=keys,
                        )
                    ]
                )
        cued = [
            s for s in allowed if words & _CUES[s] and s in {k.split(":")[0] for k in referents}
        ]
        scoped = [k for k in referents if not cued or k.split(":", 1)[0] in cued]
        picked = [k for k in _pick(question, scoped) if k.split(":", 1)[0] in allowed]
        if not picked:
            return None
        picked_sources = {k.split(":", 1)[0] for k in picked}
        sources: list[Source] = [s for s in allowed if s in picked_sources]
        return Plan(
            sub_questions=[
                SubQuestion(
                    id="sq1",
                    text=question[:300],
                    target_sources=sources,
                    relation="recall",
                    prior_keys=picked,
                )
            ]
        )

    def _carry_over(self, allowed: Sequence[Source], ctx: ConversationContext) -> Plan | None:
        """Re-plan last turn's unresolved sub-questions, seeding dependencies from memory."""
        renamed: dict[str, str] = {}
        subs: list[SubQuestion] = []
        for sq in ctx.unresolved:
            targets = [s for s in sq.target_sources if s in allowed]
            deps = [renamed[d] for d in sq.depends_on if d in renamed]
            prior = [] if deps else ctx.unresolved_dep_keys.get(sq.id, [])[:MAX_PRIOR_KEYS]
            if not targets or (sq.relation != "search" and not deps and not prior):
                continue
            new_id = f"sq{len(subs) + 1}"
            renamed[sq.id] = new_id
            subs.append(
                sq.model_copy(
                    update={
                        "id": new_id,
                        "target_sources": targets,
                        "depends_on": deps,
                        "prior_keys": prior,
                    }
                )
            )
        return Plan(sub_questions=subs) if subs else None


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
- Follow-ups: "prior_keys" may list record ids from "Records from the previous answer".
  relation "recall" answers from those records without searching (prior_keys required);
  "blocked_by" / "mentions" may use prior_keys instead of depends_on.
- Use only the user's questions. Never invent tool names or record ids."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_plan(text: str, sources: Sequence[Source], known_keys: Sequence[str] = ()) -> Plan:
    """Validate an LLM plan; drop disallowed sources and record ids this thread never fetched."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object in planner reply")
    raw: Any = json.loads(match.group(0))
    plan = Plan.model_validate(raw)
    known = set(known_keys)
    kept: list[SubQuestion] = []
    kept_ids: set[str] = set()
    for sq in plan.sub_questions:
        targets = [s for s in sq.target_sources if s in sources]
        prior = [k for k in sq.prior_keys if k in known]
        needs_prior = sq.relation == "recall" or (sq.relation != "search" and not sq.depends_on)
        if (
            not targets
            or any(d not in kept_ids for d in sq.depends_on)
            or (needs_prior and not prior)
        ):
            continue
        kept.append(sq.model_copy(update={"target_sources": targets, "prior_keys": prior}))
        kept_ids.add(sq.id)
    if not kept:
        raise ValueError("plan has no sub-question for the allowed sources")
    return Plan(sub_questions=kept)


def context_lines(ctx: ConversationContext) -> str:
    """Planner-visible memory: user questions, opaque record ids, unresolved sub-questions."""
    parts = ["Earlier questions in this conversation:"]
    parts += [f"- {q}" for q in ctx.previous_questions]
    if ctx.referents:
        parts.append("Records from the previous answer (ids only, in order):")
        parts += [f"{i}. {k}" for i, k in enumerate(ctx.referents, start=1)]
    if ctx.unresolved:
        parts.append("Unanswered sub-questions from the previous turn:")
        parts += [
            f"- {sq.id} ({sq.relation} -> {','.join(sq.target_sources)}): {sq.text}"
            for sq in ctx.unresolved
        ]
    return "\n".join(parts)


class LLMPlanner:
    def __init__(self, llm: BaseChatModel, fallback: Planner | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RulePlanner()

    def plan(
        self,
        question: str,
        sources: Sequence[Source],
        context: ConversationContext | None = None,
    ) -> tuple[Plan, list[str], TokenUsage]:
        user = f"Question: {question}\nAllowed sources: {', '.join(sources)}"
        if context is not None:
            user += "\n\n" + context_lines(context)
        messages = [SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(user)]
        reply = self.llm.invoke(messages)
        usage: Mapping[str, Any] = (
            (reply.usage_metadata or {}) if isinstance(reply, AIMessage) else {}
        )
        tokens = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        known = context.known_keys if context is not None else []
        try:
            return parse_plan(text, sources, known), [], tokens
        except (ValueError, ValidationError) as exc:
            plan, notes, _ = self.fallback.plan(question, sources, context)
            reason = type(exc).__name__
            return plan, [*notes, f"planner: invalid LLM plan ({reason}); used rule plan"], tokens


def make_planner(provider: str, llm: BaseChatModel) -> Planner:
    return RulePlanner() if provider == "mock" else LLMPlanner(llm)
