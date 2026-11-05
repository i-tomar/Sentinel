"""Documents (one per source record) and the chunks stored in the vector index."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["notion", "jira", "gmail"]
SOURCES: tuple[Source, ...] = ("notion", "jira", "gmail")


def doc_key(source: str, doc_id: str) -> str:
    """Stable cross-source identifier, e.g. `jira:PROJ-142` or `notion:auth-migration-v2`."""
    return f"{source}:{doc_id}"


class Document(BaseModel):
    """One source record, normalized. `text` is untrusted data, never instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Source
    id: str
    url: str
    title: str
    text: str
    updated_at: datetime
    """Last-modified time reported by the source (not by Sentinel)."""
    fetched_at: datetime
    """When Sentinel last pulled this record."""
    author: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return doc_key(self.source, self.id)

    def content_hash(self) -> str:
        """Hash of everything the source controls; `fetched_at` is excluded on purpose."""
        payload = self.model_dump(mode="json", exclude={"fetched_at"})
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_key: str
    source: Source
    doc_id: str
    title: str
    url: str
    updated_at: datetime
    fetched_at: datetime
    index: int
    text: str
    doc_hash: str


SearchMode = Literal["hybrid", "dense", "keyword"]


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
