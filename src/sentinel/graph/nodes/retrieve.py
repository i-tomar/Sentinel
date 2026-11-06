"""Index retrieval with a lexical relevance gate, and evidence assembly for the answerer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sentinel.citations import CitationResolver
from sentinel.graph.state import Evidence, SubQuestion
from sentinel.guards.sanitize import mask_neutralized
from sentinel.retrieval.models import Chunk, Document, SearchHit
from sentinel.text import content_tokens

MAX_EVIDENCE = 12
MAX_EVIDENCE_CHARS = 1500


def is_relevant(question: str, chunk: Chunk) -> bool:
    """Dense search always returns *something*; keep a chunk only if it shares at least one
    content word with the question. Crude, but it lets the agent say "no evidence" honestly.
    Words inside text the guard quoted as untrusted do not count."""
    visible = mask_neutralized(f"{chunk.title}\n{chunk.text}")
    return bool(content_tokens(question) & content_tokens(visible))


def relevant_chunks(question: str, hits: Sequence[SearchHit]) -> list[Chunk]:
    return [h.chunk for h in hits if is_relevant(question, h.chunk)]


def assemble_evidence(
    question: str,
    plan: Sequence[SubQuestion],
    documents: Mapping[str, Document],
    sq_docs: Mapping[str, Sequence[str]],
    index_chunks: Iterable[Chunk],
    stale_keys: Iterable[str],
    *,
    extra_buckets: Sequence[str] = (),
    guarded: Mapping[str, int] | None = None,
    max_items: int = MAX_EVIDENCE,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> list[Evidence]:
    """Records fetched by tools come first, in plan order (fresh data, targeted lookups),
    then records in `extra_buckets` (e.g. re-fetched sources of memory notes), then relevant
    index chunks. One numbered evidence item per record."""
    stale = set(stale_keys)
    resolver = CitationResolver()
    items: dict[int, Evidence] = {}

    def put(n: int, key: str, text: str) -> None:
        if n in items:
            if len(items[n].text) < max_chars:
                merged = f"{items[n].text}\n...\n{text}"[:max_chars]
                items[n] = items[n].model_copy(update={"text": merged})
            return
        if len(items) >= max_items:
            return
        citation = resolver.resolve(n)
        if citation is None:  # cannot happen: n was just issued by the resolver
            raise RuntimeError(f"citation {n} vanished")
        items[n] = Evidence(
            citation=citation,
            text=text[:max_chars],
            score=1.0,
            stale=key in stale,
            untrusted_removed=(guarded or {}).get(key, 0),
        )

    for bucket in [*(sq.id for sq in plan), *extra_buckets]:
        for key in sq_docs.get(bucket, []):
            doc = documents.get(key)
            if doc is not None and len(items) < max_items:
                put(resolver.add_document(doc).n, key, doc.text)
    for chunk in index_chunks:
        if not is_relevant(question, chunk):
            continue
        if len(items) >= max_items and resolver.resolve_key(chunk.doc_key) is None:
            continue
        put(resolver.add(chunk).n, chunk.doc_key, chunk.text)
    return [items[n] for n in sorted(items)]
