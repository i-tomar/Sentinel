"""Decide after each hop: route more sub-questions, widen an empty search, or answer.

Also marks stale records: when several Notion pages cover the same topic (titles equal once
version/year markers are removed), all but the most recently updated are stale.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from sentinel.graph.nodes.tool_router import ready
from sentinel.graph.state import SubQuestion
from sentinel.retrieval.models import Chunk, Document, Source

_VERSION_MARKERS = re.compile(r"\(\s*v\d+\s*\)|\bv\d+\b|\b(?:19|20)\d{2}\b", re.I)


def topic_key(title: str) -> str:
    return " ".join(_VERSION_MARKERS.sub(" ", title).lower().split())


def find_stale(records: Iterable[Document | Chunk]) -> list[str]:
    newest: dict[str, tuple[str, float]] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.source != "notion":
            continue
        key = r.key if isinstance(r, Document) else r.doc_key
        topic = topic_key(r.title)
        groups[topic].add(key)
        ts = r.updated_at.timestamp()
        if topic not in newest or ts > newest[topic][1]:
            newest[topic] = (key, ts)
    stale = {k for topic, keys in groups.items() for k in keys if k != newest[topic][0]}
    return sorted(stale)


def decide(
    plan: Sequence[SubQuestion],
    done: Sequence[str],
    *,
    hops: int,
    max_hops: int,
    has_evidence: Mapping[str, bool],
    widened: Sequence[str],
    allowed_sources: Sequence[Source],
    tools_available: bool,
) -> tuple[Literal["route", "answer"], list[str], list[str]]:
    """Return (next step, sub-questions to widen, unanswered sub-question ids).

    A search sub-question that found nothing is retried once over the allowed sources it
    did not target; its filters (time window, label) are kept, so the user's constraints hold.
    """
    unanswered = [sq.id for sq in plan if sq.id not in done or not has_evidence.get(sq.id)]
    if hops >= max_hops:
        return "answer", [], unanswered
    if ready(plan, done):
        return "route", [], unanswered
    widen = [
        sq.id
        for sq in plan
        if tools_available
        and sq.relation == "search"
        and sq.id in done
        and sq.id not in widened
        and not has_evidence.get(sq.id)
        and set(allowed_sources) - set(sq.target_sources)
    ]
    if widen:
        return "route", widen, unanswered
    return "answer", [], unanswered
