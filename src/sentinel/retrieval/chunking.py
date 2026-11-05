"""Split documents into overlapping, line-aligned chunks with deterministic IDs."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sentinel.retrieval.models import Chunk, Document

CHUNK_NAMESPACE = uuid.UUID("5b0f3c9e-6a52-4c1e-9d4f-3f1d2a7c8e10")
DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP_CHARS = 120


def chunk_id(key: str, index: int) -> str:
    """UUIDv5, so re-ingesting a document overwrites its points instead of duplicating them."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{key}#{index}"))


def _pack(items: Iterable[str], sep: str, max_chars: int, overlap: int) -> list[str]:
    """Greedily join items up to `max_chars`; each new piece re-uses trailing items that fit
    in `overlap`. Every item must already be <= max_chars, so every piece is too."""
    out: list[str] = []
    current: list[str] = []
    for item in items:
        if current and len(sep.join([*current, item])) > max_chars:
            out.append(sep.join(current))
            carry: list[str] = []
            for prev in reversed(current):
                if len(sep.join([prev, *carry])) > overlap:
                    break
                carry.insert(0, prev)
            current = carry if len(sep.join([*carry, item])) <= max_chars else []
        current.append(item)
    if current:
        out.append(sep.join(current))
    return out


def _wrap_line(line: str, max_chars: int, overlap: int) -> list[str]:
    words = [w[i : i + max_chars] for w in line.split() for i in range(0, len(w), max_chars)]
    return _pack(words, " ", max_chars, overlap)


def split_text(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP_CHARS
) -> list[str]:
    """Pack whole lines into chunks of at most `max_chars`; over-long lines are word-wrapped
    (and over-long words, e.g. base64 blobs, hard-split)."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("need max_chars > 0 and 0 <= overlap < max_chars")
    lines: list[str] = []
    for line in text.splitlines():
        lines.extend(_wrap_line(line, max_chars, overlap) if len(line) > max_chars else [line])
    return [c.strip() for c in _pack(lines, "\n", max_chars, overlap) if c.strip()]


def chunk_document(
    doc: Document, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP_CHARS
) -> list[Chunk]:
    doc_hash = doc.content_hash()
    pieces = split_text(doc.text, max_chars, overlap) or [doc.title]
    return [
        Chunk(
            chunk_id=chunk_id(doc.key, i),
            doc_key=doc.key,
            source=doc.source,
            doc_id=doc.id,
            title=doc.title,
            url=doc.url,
            updated_at=doc.updated_at,
            fetched_at=doc.fetched_at,
            index=i,
            text=piece,
            doc_hash=doc_hash,
        )
        for i, piece in enumerate(pieces)
    ]
