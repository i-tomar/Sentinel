"""Memory items and memory write requests."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel.retrieval.embed import keyword_tokens
from sentinel.text import STOPWORDS

MEMORY_KEY_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,99}$"
MEMORY_VALUE_MAX = 500
MEMORY_SOURCE_PATTERN = r"^(?:(?:notion|jira|gmail):[\w.-]{1,100}|user|mcp|agent)$"
"""A record key the fact was taken from, or who asserted it (user, mcp client, agent)."""


class MemoryItem(BaseModel):
    """One remembered fact. A hint for the agent, never citable evidence by itself."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=MEMORY_KEY_PATTERN)
    value: str = Field(min_length=1, max_length=MEMORY_VALUE_MAX)
    source: str = Field(pattern=MEMORY_SOURCE_PATTERN)
    verified: bool = False
    """True only when the agent checked the value against its source record."""
    stale: bool = False
    updated_at: datetime
    """When the fact was true at its source (the source record's updated_at for record facts)."""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    record_keys: list[str] = Field(default_factory=list, max_length=8)
    """All source records that supported the value (source is the primary one)."""
    stale_reason: str | None = Field(default=None, max_length=300)
    superseded_by: str | None = None

    @property
    def source_record(self) -> str | None:
        return self.source if ":" in self.source else None


class MemoryWrite(BaseModel):
    """A requested change to long-term memory and what happened to it."""

    op: Literal["put", "mark_stale"]
    key: str
    status: Literal["written", "pending_approval", "skipped"]
    reason: str
    item: MemoryItem | None = None


def memory_key(text: str) -> str:
    """Stable key from the content words of a question or topic: "Who owns the Q3 plan?" ->
    "owns-q3-plan"."""
    words = [t for t in keyword_tokens(text) if t not in STOPWORDS and "-" not in t]
    key = "-".join(dict.fromkeys(words))[:100].strip("-")
    key = re.sub(r"[^a-z0-9._-]", "", key)
    return key or "untitled"
