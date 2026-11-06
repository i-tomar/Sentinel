"""Short-term memory: what the planner may know about earlier turns of a thread.

The context holds the user's earlier questions, the previous turn's unresolved sub-questions,
and opaque record keys (`jira:PROJ-142`) in citation order. It never holds record text, so
retrieved content still cannot steer planning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from sentinel.graph.state import SubQuestion, TurnRecord
from sentinel.retrieval.models import Document

MAX_TURNS = 8
MAX_REFERENTS = 12


class ConversationContext(BaseModel):
    previous_questions: list[str]
    referents: list[str] = Field(default_factory=list)
    """What "them" / "the second one" refer to: the records found by the last turn's final
    answered sub-question (its answer set); else its cited records; else its evidence."""
    unresolved: list[SubQuestion] = Field(default_factory=list)
    """Last turn's sub-questions that ended without evidence (hop limit, failures, ...)."""
    unresolved_dep_keys: dict[str, list[str]] = Field(default_factory=dict)
    """Unresolved sub-question id -> records its dependencies found last turn."""
    known_keys: list[str] = Field(default_factory=list)


def context_from(
    turns: Sequence[TurnRecord], documents: Mapping[str, Document]
) -> ConversationContext | None:
    if not turns:
        return None
    last = turns[-1]
    answer_set: list[str] = next(
        (last.sq_docs[sq.id] for sq in reversed(last.plan) if last.sq_docs.get(sq.id)), []
    )
    candidates = answer_set or last.cited_keys or last.evidence_keys
    referents = [k for k in dict.fromkeys(candidates) if k in documents][:MAX_REFERENTS]
    unresolved = [sq for sq in last.plan if sq.id in last.unanswered]
    dep_keys = {
        sq.id: [k for d in sq.depends_on for k in last.sq_docs.get(d, []) if k in documents]
        + [k for k in sq.prior_keys if k in documents]
        for sq in unresolved
    }
    return ConversationContext(
        previous_questions=[t.question for t in turns][-3:],
        referents=referents,
        unresolved=unresolved,
        unresolved_dep_keys=dep_keys,
        known_keys=sorted(documents),
    )


def prune_memory(
    turns: Sequence[TurnRecord], documents: Mapping[str, Document], max_turns: int = MAX_TURNS
) -> tuple[list[TurnRecord], dict[str, Document]]:
    """Keep the last `max_turns` turns and only the records they reference."""
    kept = list(turns)[-max_turns:]
    referenced = {
        k
        for t in kept
        for k in (*t.evidence_keys, *t.cited_keys, *(k for ks in t.sq_docs.values() for k in ks))
    }
    return kept, {k: d for k, d in documents.items() if k in referenced}
