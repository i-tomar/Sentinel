"""Freshness rule: when memory conflicts with a newer source record, the source wins.

A remembered fact conflicts with its source when
- its source record has been superseded by a newer record on the same topic, or
- its source record changed after the fact was taken and no longer supports the fact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel

from sentinel.memory.models import MemoryItem
from sentinel.retrieval.models import Document
from sentinel.text import content_tokens

SUPPORT_THRESHOLD = 0.6


def _claim_words(text: str) -> set[str]:
    """Content words minus record ids ("PROJ-142") and their parts: ids say what a fact is
    about, not what it claims, and would otherwise dominate the overlap."""
    words = content_tokens(text)
    ids = {w for w in words if "-" in w and any(c.isdigit() for c in w)}
    parts = {p for w in ids for p in w.split("-")}
    return (words - ids - parts) or words


def support(value: str, text: str) -> float:
    """Share of the value's claim words present in `text`."""
    words = _claim_words(value)
    if not words:
        return 0.0
    return len(words & content_tokens(text)) / len(words)


class Conflict(BaseModel):
    key: str
    reason: str
    superseded_by: str | None


def find_conflicts(
    items: Iterable[MemoryItem],
    documents: Mapping[str, Document],
    stale_record_keys: Sequence[str],
    newest_by_stale: Mapping[str, str] | None = None,
) -> list[Conflict]:
    stale_records = set(stale_record_keys)
    conflicts: list[Conflict] = []
    for item in items:
        record = item.source_record
        if item.stale or record is None:
            continue
        if record in stale_records:
            newer = (newest_by_stale or {}).get(record)
            conflicts.append(
                Conflict(
                    key=item.key,
                    reason=f"source record {record} was superseded by a newer record",
                    superseded_by=newer,
                )
            )
            continue
        doc = documents.get(record)
        if doc is None or doc.updated_at <= item.updated_at:
            continue
        if support(item.value, f"{doc.title}\n{doc.text}") < SUPPORT_THRESHOLD:
            conflicts.append(
                Conflict(
                    key=item.key,
                    reason=(
                        f"source record {record} changed on {doc.updated_at.date()} "
                        "and no longer supports the remembered value"
                    ),
                    superseded_by=record,
                )
            )
    return conflicts
