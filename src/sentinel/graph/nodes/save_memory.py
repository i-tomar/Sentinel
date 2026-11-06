"""Propose long-term memory writes after an answer, and apply them under the write policy.

Two kinds of write:
- mark_stale for notes that conflicted with a newer source record this turn;
- put of the answer as a verified fact, only when every sentence passed the citation check
  against records fetched this turn (no unknown, uncited, unsupported or memory-only claims).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from sentinel.citations import Citation, cited_numbers
from sentinel.graph.nodes.answerer import is_insufficient
from sentinel.graph.state import CitationReport, Evidence
from sentinel.llm import INSUFFICIENT_PREFIX, PARTIAL_PREFIX
from sentinel.memory.backend import MemoryBackend
from sentinel.memory.freshness import support
from sentinel.memory.models import MemoryItem, MemoryWrite, memory_key
from sentinel.memory.policy import WriteDecision
from sentinel.text import split_sentences

MAX_VALUE_CHARS = 500
_MARKERS = re.compile(r"\s*\[\d{1,3}\]")


def fact_from_answer(
    question: str,
    answer: str,
    citations: Sequence[Citation],
    report: CitationReport,
    evidence: Sequence[Evidence],
) -> MemoryItem | None:
    if is_insufficient(answer) or not citations or not report.ok:
        return None
    sentences = [
        s
        for s in split_sentences(answer)
        if cited_numbers(s) and not s.startswith((PARTIAL_PREFIX, INSUFFICIENT_PREFIX))
    ]
    if not sentences:
        return None
    value = " ".join(_MARKERS.sub("", s) for s in sentences)
    value = " ".join(value.split())[:MAX_VALUE_CHARS].strip()
    if not value:
        return None
    by_key = {ev.citation.key: ev for ev in evidence}
    primary = citations[0]
    cited_text = "\n".join(by_key[c.key].text for c in citations if c.key in by_key)
    return MemoryItem(
        key=memory_key(question),
        value=value,
        source=primary.key,
        verified=True,
        stale=False,
        updated_at=primary.updated_at,
        confidence=round(min(1.0, support(value, cited_text)), 3),
        record_keys=[c.key for c in citations][:8],
    )


def propose_writes(
    conflicting: Sequence[MemoryItem], fact: MemoryItem | None, existing: MemoryItem | None
) -> list[MemoryWrite]:
    writes = [
        MemoryWrite(
            op="mark_stale",
            key=n.key,
            status="pending_approval",
            reason=n.stale_reason or "conflicts with a newer source record",
            item=n,
        )
        for n in conflicting
    ]
    if fact is not None:
        unchanged = (
            existing is not None
            and not existing.stale
            and existing.value == fact.value
            and existing.source == fact.source
        )
        writes.append(
            MemoryWrite(
                op="put",
                key=fact.key,
                status="skipped" if unchanged else "pending_approval",
                reason="already remembered" if unchanged else "verified answer from cited records",
                item=fact,
            )
        )
    return writes


def apply_writes(
    writes: Sequence[MemoryWrite], backend: MemoryBackend | None, decision: WriteDecision
) -> list[MemoryWrite]:
    """Execute allowed writes; leave the rest pending (approval flow arrives with step l)."""
    applied: list[MemoryWrite] = []
    for w in writes:
        if w.status != "pending_approval" or backend is None or decision != "allow":
            applied.append(w)
            continue
        if w.op == "put" and w.item is not None:
            backend.put(w.item)
        elif w.op == "mark_stale" and w.item is not None:
            backend.mark_stale(w.key, w.reason, w.item.superseded_by)
        applied.append(w.model_copy(update={"status": "written"}))
    return applied


def flag_conflicts(
    notes: Sequence[MemoryItem], conflicts: Mapping[str, tuple[str, str | None]]
) -> list[MemoryItem]:
    """Notes with this turn's conflicts applied (stale, reason, superseding record)."""
    return [
        n.model_copy(
            update={
                "stale": True,
                "stale_reason": conflicts[n.key][0],
                "superseded_by": conflicts[n.key][1],
            }
        )
        if n.key in conflicts
        else n
        for n in notes
    ]
