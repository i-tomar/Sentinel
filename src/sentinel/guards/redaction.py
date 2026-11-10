"""Apply PII redaction to the objects that carry text through Sentinel.

Records are redacted when they enter agent state, so the prompt, the answer, traces and both
memories only ever see the redacted copy. Identifiers and URLs are left alone: citations must
keep resolving (the detector has no URL recognizer and does not fire on ticket/page/thread ids).
"""

from __future__ import annotations

from functools import lru_cache

from sentinel.citations import Citation
from sentinel.guards.pii import redact
from sentinel.memory.models import MemoryItem
from sentinel.retrieval.models import Chunk, Document


@lru_cache(maxsize=4096)
def _redact(text: str) -> tuple[str, tuple[str, ...]]:
    clean, findings = redact(text)
    return clean, tuple(f.kind for f in findings)


def redact_str(text: str) -> str:
    """Cached: the same record text is redacted once per process, not once per hop."""
    return _redact(text)[0] if text else text


def redaction_kinds(text: str) -> tuple[str, list[str]]:
    clean, kinds = _redact(text) if text else (text, ())
    return clean, sorted(set(kinds))


def redact_document(doc: Document) -> Document:
    return doc.model_copy(
        update={
            "title": redact_str(doc.title),
            "text": redact_str(doc.text),
            "author": redact_str(doc.author) if doc.author else doc.author,
            "metadata": {k: redact_str(v) for k, v in doc.metadata.items()},
        }
    )


def redact_chunk(chunk: Chunk) -> Chunk:
    return chunk.model_copy(
        update={"title": redact_str(chunk.title), "text": redact_str(chunk.text)}
    )


def redact_citation(citation: Citation) -> Citation:
    return citation.model_copy(
        update={"title": redact_str(citation.title), "snippet": redact_str(citation.snippet)}
    )


def redact_memory_item(item: MemoryItem) -> MemoryItem:
    return item.model_copy(
        update={
            "value": redact_str(item.value),
            "stale_reason": redact_str(item.stale_reason) if item.stale_reason else None,
        }
    )
