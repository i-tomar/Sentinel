"""MemoryBackend protocol and backend-independent search ranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from sentinel.memory.models import MemoryItem
from sentinel.text import content_tokens


class MemoryBackend(Protocol):
    def get(self, key: str) -> MemoryItem | None: ...

    def put(self, item: MemoryItem) -> MemoryItem:
        """Insert or replace the item with the same key."""
        ...

    def search(
        self, query: str, limit: int = 5, *, include_stale: bool = False
    ) -> list[MemoryItem]: ...

    def mark_stale(
        self, key: str, reason: str, superseded_by: str | None = None
    ) -> MemoryItem | None:
        """Flag an item as stale; returns the updated item, or None if the key is unknown."""
        ...

    def all(self, *, include_stale: bool = True) -> list[MemoryItem]: ...


def relevance(query: str, item: MemoryItem) -> int:
    return len(content_tokens(query) & content_tokens(f"{item.key.replace('-', ' ')} {item.value}"))


def rank(
    query: str, items: Iterable[MemoryItem], limit: int, include_stale: bool
) -> list[MemoryItem]:
    """Matching items, fresh before stale and verified before unverified, then by relevance,
    confidence and recency."""
    scored = [(relevance(query, i), i) for i in items if include_stale or not i.stale]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(
        key=lambda si: (
            si[1].stale,
            not si[1].verified,
            -si[0],
            -si[1].confidence,
            -si[1].updated_at.timestamp(),
            si[1].key,
        )
    )
    return [i for _, i in scored[:limit]]


def keys(items: Sequence[MemoryItem]) -> list[str]:
    return [i.key for i in items]
