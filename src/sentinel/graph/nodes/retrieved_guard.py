"""Scan every retrieved record and index chunk before it can reach planning or the prompt.

Runs after each hop's tool calls and index search. Records are sanitized in place: later hops
only see neutralized text (so injected ticket keys cannot become tool arguments), and so does
the answerer. Already-neutralized spans are masked before re-scanning, so a later hop neither
re-wraps quoted text nor reports the same record twice.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Literal

from sentinel.graph.nodes.tool_router import MEMORY_BUCKET
from sentinel.graph.state import GuardNote
from sentinel.guards.injection import InjectionDetector, InjectionResult
from sentinel.guards.sanitize import GuardMode, mask_neutralized, neutralize
from sentinel.memory.models import MEMORY_VALUE_MAX, MemoryItem
from sentinel.retrieval.models import Chunk, Document
from sentinel.retrieval.normalize import strip_field_labels
from sentinel.text import content_tokens

TITLE_REMOVED = "[title removed: suspected injected instruction]"


def _scan(detector: InjectionDetector, text: str) -> InjectionResult:
    return detector.scan(mask_neutralized(text), origin="retrieved")


def _action(mode: GuardMode) -> Literal["removed", "quoted"]:
    return "removed" if mode == "remove" else "quoted"


def guard_document(
    doc: Document, detector: InjectionDetector, mode: GuardMode
) -> tuple[Document, GuardNote | None]:
    title_result = _scan(detector, doc.title)
    text_result = _scan(detector, doc.text)
    if not (title_result.flagged or text_result.flagged):
        return doc, None
    text, spans = neutralize(doc.text, text_result, mode)
    note = GuardNote(
        key=doc.key,
        score=max(title_result.score, text_result.score),
        rules=sorted({f.rule for f in [*title_result.findings, *text_result.findings]}),
        spans=spans + int(title_result.flagged),
        action=_action(mode),
    )
    title = TITLE_REMOVED if title_result.flagged else doc.title
    return doc.model_copy(update={"title": title, "text": text}), note


def guard_chunk(
    chunk: Chunk, detector: InjectionDetector, mode: GuardMode
) -> tuple[Chunk, GuardNote | None]:
    title_result = _scan(detector, chunk.title)
    text_result = _scan(detector, chunk.text)
    if not (title_result.flagged or text_result.flagged):
        return chunk, None
    text, spans = neutralize(chunk.text, text_result, mode)
    note = GuardNote(
        key=chunk.doc_key,
        score=max(title_result.score, text_result.score),
        rules=sorted({f.rule for f in [*title_result.findings, *text_result.findings]}),
        spans=spans + int(title_result.flagged),
        action=_action(mode),
    )
    title = TITLE_REMOVED if title_result.flagged else chunk.title
    return chunk.model_copy(update={"title": title, "text": text}), note


def guard_memory(
    notes: Sequence[MemoryItem], detector: InjectionDetector, mode: GuardMode
) -> tuple[list[MemoryItem], list[GuardNote]]:
    """Long-term memory comes back from a tool (the memory server) too, so it is data as well."""
    clean: list[MemoryItem] = []
    flagged: list[GuardNote] = []
    for item in notes:
        result = _scan(detector, item.value)
        if not result.flagged:
            clean.append(item)
            continue
        value, spans = neutralize(item.value, result, mode)
        clean.append(item.model_copy(update={"value": value[:MEMORY_VALUE_MAX]}))
        flagged.append(
            GuardNote(
                key=f"{MEMORY_BUCKET}:{item.key}",
                score=result.score,
                rules=sorted({f.rule for f in result.findings}),
                spans=spans,
                action=_action(mode),
            )
        )
    return clean, flagged


def guard_all(
    documents: Mapping[str, Document],
    chunks: Sequence[Chunk],
    detector: InjectionDetector,
    mode: GuardMode,
) -> tuple[dict[str, Document], list[Chunk], list[GuardNote]]:
    notes: list[GuardNote] = []
    clean_docs: dict[str, Document] = {}
    for key, doc in documents.items():
        clean_docs[key], note = guard_document(doc, detector, mode)
        if note:
            notes.append(note)
    clean_chunks: list[Chunk] = []
    for chunk in chunks:
        clean, note = guard_chunk(chunk, detector, mode)
        clean_chunks.append(clean)
        if note:
            notes.append(note)
    return clean_docs, clean_chunks, notes


def recheck_relevance(
    sq_docs: Mapping[str, Sequence[str]],
    documents: Mapping[str, Document],
    queries: Mapping[str, Sequence[str]],
    guarded_keys: Collection[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Drop a sanitized record from a sub-question's results when it matched that
    sub-question's search queries only through the text the guard neutralized.

    Otherwise injected words could pull a record into the results, and its own id would then
    drive the next hop. Records found by structured filters or direct fetches (no query) stay.
    Returns (sq_docs, dropped "sub_question_id:key" pairs).
    """
    kept: dict[str, list[str]] = {}
    dropped: list[str] = []
    for sq_id, keys in sq_docs.items():
        query_tokens = set().union(*(content_tokens(q) for q in queries.get(sq_id, ())))
        kept[sq_id] = []
        for key in keys:
            doc = documents.get(key)
            if query_tokens and key in guarded_keys and doc is not None:
                visible = content_tokens(
                    strip_field_labels(mask_neutralized(f"{doc.title}\n{doc.text}"))
                )
                if not query_tokens & visible:
                    dropped.append(f"{sq_id}:{key}")
                    continue
            kept[sq_id].append(key)
    return kept, dropped


def merge_notes(existing: Sequence[GuardNote], new: Sequence[GuardNote]) -> list[GuardNote]:
    """Append new notes, dropping exact repeats (a record re-fetched raw on a later hop)."""
    merged = list(existing)
    for note in new:
        if note not in merged:
            merged.append(note)
    return merged
