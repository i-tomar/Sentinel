"""Citations: numbered references to evidence that was actually retrieved.

`CitationResolver` only knows chunks that were registered from real search/tool results, so an
answer marker like `[7]` either resolves to a returned record or is reported as unknown; it can
never be satisfied by a fabricated source.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from sentinel.retrieval.models import Chunk, Document, Source, doc_key

SNIPPET_CHARS = 280
_MARKER = re.compile(r"\[(\d{1,3})\]")
_SOURCE_LABEL: dict[Source, str] = {"notion": "Notion", "jira": "Jira", "gmail": "Gmail"}


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(ge=1)
    source: Source
    doc_id: str
    title: str
    url: str
    snippet: str
    updated_at: datetime
    fetched_at: datetime

    @property
    def key(self) -> str:
        return doc_key(self.source, self.doc_id)

    def render(self) -> str:
        return (
            f"[{self.n}] {_SOURCE_LABEL[self.source]}: {self.title} ({self.doc_id}) {self.url} "
            f"- updated {self.updated_at.date().isoformat()}, "
            f"fetched {self.fetched_at.isoformat(timespec='seconds')}"
        )


def cited_numbers(text: str) -> list[int]:
    """Distinct `[n]` markers in order of first appearance."""
    return list(dict.fromkeys(int(m) for m in _MARKER.findall(text)))


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 3].rstrip() + "..."


class CitationCheck(BaseModel):
    used: list[int]
    unknown: list[int]
    """Markers with no registered evidence (hallucinated citations)."""
    unused: list[int]

    @property
    def ok(self) -> bool:
        return not self.unknown


class CitationResolver:
    """Assigns stable numbers per document (`[n]` is per record, not per chunk)."""

    def __init__(self) -> None:
        self._by_key: dict[str, Citation] = {}
        self._by_n: dict[int, Citation] = {}

    def add(self, chunk: Chunk) -> Citation:
        """Register a retrieved index chunk (numbered per document)."""
        return self._register(
            chunk.source,
            chunk.doc_id,
            chunk.title,
            chunk.url,
            chunk.text,
            chunk.updated_at,
            chunk.fetched_at,
        )

    def add_document(self, doc: Document) -> Citation:
        """Register a record returned directly by an MCP tool call."""
        return self._register(
            doc.source, doc.id, doc.title, doc.url, doc.text, doc.updated_at, doc.fetched_at
        )

    def _register(
        self,
        source: Source,
        doc_id: str,
        title: str,
        url: str,
        text: str,
        updated_at: datetime,
        fetched_at: datetime,
    ) -> Citation:
        key = doc_key(source, doc_id)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        citation = Citation(
            n=len(self._by_n) + 1,
            source=source,
            doc_id=doc_id,
            title=title,
            url=url,
            snippet=_snippet(text),
            updated_at=updated_at,
            fetched_at=fetched_at,
        )
        self._by_key[key] = citation
        self._by_n[citation.n] = citation
        return citation

    def resolve(self, n: int) -> Citation | None:
        return self._by_n.get(n)

    def resolve_key(self, key: str) -> Citation | None:
        return self._by_key.get(key)

    @property
    def citations(self) -> list[Citation]:
        return [self._by_n[n] for n in sorted(self._by_n)]

    def check(self, answer: str) -> CitationCheck:
        used = cited_numbers(answer)
        return CitationCheck(
            used=used,
            unknown=[n for n in used if n not in self._by_n],
            unused=[n for n in sorted(self._by_n) if n not in used],
        )

    def render(self, numbers: list[int] | None = None) -> str:
        chosen = numbers if numbers is not None else sorted(self._by_n)
        return "\n".join(self._by_n[n].render() for n in chosen if n in self._by_n)
