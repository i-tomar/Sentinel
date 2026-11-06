"""Hybrid search, then a lexical relevance gate, then citation numbering per record."""

from __future__ import annotations

from collections.abc import Sequence

from sentinel.citations import CitationResolver
from sentinel.graph.state import Evidence
from sentinel.retrieval.models import SearchHit
from sentinel.text import content_tokens


def is_relevant(question: str, hit: SearchHit) -> bool:
    """Dense search always returns *something*; keep a hit only if it shares at least one
    content word with the question. Crude, but it lets the agent say "no evidence" honestly."""
    return bool(content_tokens(question) & content_tokens(f"{hit.chunk.title}\n{hit.chunk.text}"))


def build_evidence(question: str, hits: Sequence[SearchHit]) -> list[Evidence]:
    resolver = CitationResolver()
    merged: dict[int, Evidence] = {}
    for hit in hits:
        if not is_relevant(question, hit):
            continue
        citation = resolver.add(hit.chunk)
        if citation.n in merged:
            prev = merged[citation.n]
            merged[citation.n] = prev.model_copy(
                update={"text": f"{prev.text}\n...\n{hit.chunk.text}"}
            )
        else:
            merged[citation.n] = Evidence(citation=citation, text=hit.chunk.text, score=hit.score)
    return [merged[n] for n in sorted(merged)]
