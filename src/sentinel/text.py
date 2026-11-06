"""Small text helpers shared by retrieval gating, the mock LLM and the citation checker."""

from __future__ import annotations

import re

from sentinel.retrieval.embed import keyword_tokens

# fmt: off
STOPWORDS = frozenset((
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "before", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of",
    "on", "or", "our", "out", "per", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your",
))
# fmt: on
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


_SUFFIXES = ("ing", "ers", "er", "ed", "es", "s")


def stem(token: str) -> str:
    """Crude suffix stripping so "owns", "owner" and "owned" compare equal. Applied to both
    sides of every comparison, so consistency matters more than linguistic accuracy."""
    if token.isalpha():
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> set[str]:
    """Stemmed keyword tokens minus stopwords and one-letter noise."""
    return {
        stem(t) for t in keyword_tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
    }


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation or line breaks; drops empty pieces."""
    pieces: list[str] = []
    for line in text.splitlines():
        pieces.extend(p.strip() for p in _SENTENCE_END.split(line) if p.strip())
    return pieces
