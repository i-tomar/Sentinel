"""Turn ready sub-questions into read-only tool calls.

Only (source, tool) pairs in `READ_TOOLS` can be produced. Dependent sub-questions take
arguments from earlier results only as ticket keys matching a strict pattern, capped at
MAX_ENTITIES, so text inside records cannot inject instructions or pick tools.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta

from sentinel.graph.state import SubQuestion, TimeWindow, ToolCall
from sentinel.mcp_clients.tools import READ_TOOLS, SOURCE_TOOLS
from sentinel.retrieval.models import Document, Source

SEARCH_LIMIT = 5
SEARCH_EXPAND = 3
LINKED_LIMIT = 10
MAX_ENTITIES = 6
_TICKET_KEY = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")


def window_dates(window: TimeWindow, as_of: datetime) -> tuple[date, date]:
    """[after, before) dates. Weeks start on Monday."""
    today = as_of.date()
    week_start = today - timedelta(days=today.weekday())
    if window == "last_week":
        return week_start - timedelta(days=7), week_start
    if window == "this_week":
        return week_start, today + timedelta(days=1)
    return today - timedelta(days=7), today + timedelta(days=1)


def ticket_keys(docs: Iterable[Document]) -> list[str]:
    """Ticket keys in order of appearance: a Jira record's own key first, then mentions."""
    keys: list[str] = []
    for doc in docs:
        if doc.source == "jira":
            keys.append(doc.id)
        keys.extend(_TICKET_KEY.findall(f"{doc.title}\n{doc.text}"))
    return list(dict.fromkeys(keys))


def ready(plan: Sequence[SubQuestion], done: Sequence[str]) -> list[SubQuestion]:
    return [sq for sq in plan if sq.id not in done and all(d in done for d in sq.depends_on)]


def _gmail_filters(sq: SubQuestion, as_of: datetime) -> dict[str, str | int]:
    args: dict[str, str | int] = {}
    if sq.time_window:
        after, before = window_dates(sq.time_window, as_of)
        args |= {"after": after.isoformat(), "before": before.isoformat()}
    if sq.label:
        args["label"] = sq.label
    return args


def route(
    sub_questions: Sequence[SubQuestion],
    *,
    hop: int,
    as_of: datetime,
    documents: Mapping[str, Document],
    sq_docs: Mapping[str, Sequence[str]],
    stale_keys: Iterable[str],
    allowed_sources: Sequence[Source],
    widened: Iterable[str] = (),
) -> list[ToolCall]:
    stale = set(stale_keys)
    widen = set(widened)
    calls: list[ToolCall] = []

    def add(sq: SubQuestion, source: Source, tool: str, args: dict[str, str | int], n: int) -> None:
        if (source, tool) not in READ_TOOLS:  # unreachable by construction; kept as a guard
            raise ValueError(f"tool {source}.{tool} is not allowed")
        calls.append(
            ToolCall(
                id=f"h{hop}-c{len(calls) + 1}",
                sub_question_id=sq.id,
                hop=hop,
                source=source,
                tool=tool,
                args=args,
                expand=n,
            )
        )

    for sq in sub_questions:
        dep_docs = [
            documents[k]
            for d in sq.depends_on
            for k in sq_docs.get(d, [])
            if k in documents and k not in stale
        ]
        if sq.relation == "search":
            sources = (
                [s for s in allowed_sources if s not in sq.target_sources]
                if sq.id in widen
                else sq.target_sources
            )
            for source in sources:
                args: dict[str, str | int] = {"query": sq.query, "limit": SEARCH_LIMIT}
                if source == "gmail":
                    args |= _gmail_filters(sq, as_of)
                add(sq, source, SOURCE_TOOLS[source].search_tool, args, SEARCH_EXPAND)
        elif sq.relation == "blocked_by":
            for key in ticket_keys(dep_docs)[:MAX_ENTITIES]:
                add(sq, "jira", "search_issues", {"blocked_by": key, "limit": LINKED_LIMIT}, 10)
        else:  # mentions
            jira_docs = [d for d in dep_docs if d.source == "jira"]
            for key in ticket_keys(jira_docs)[:MAX_ENTITIES]:
                args = {"query": key, "limit": SEARCH_LIMIT, **_gmail_filters(sq, as_of)}
                add(sq, "gmail", "search_threads", args, SEARCH_EXPAND)
    return calls
