"""Shared loading, keyword search and JSON encoding for the fixture MCP servers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, TypeAdapter

from servers.fixtures.models import GmailThread, JiraIssue, NotionPage

DATA_DIR_ENV = "SENTINEL_FIXTURE_DATA_DIR"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

Limit = Annotated[int, Field(ge=1, le=50, description="Maximum number of results")]

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


_NOTION = TypeAdapter(tuple[NotionPage, ...])
_JIRA = TypeAdapter(tuple[JiraIssue, ...])
_GMAIL = TypeAdapter(tuple[GmailThread, ...])


def _read(name: str, directory: Path | None) -> bytes:
    return ((directory or data_dir()) / name).read_bytes()


def load_notion(directory: Path | None = None) -> tuple[NotionPage, ...]:
    return _NOTION.validate_json(_read("notion_pages.json", directory))


def load_jira(directory: Path | None = None) -> tuple[JiraIssue, ...]:
    return _JIRA.validate_json(_read("jira_issues.json", directory))


def load_gmail(directory: Path | None = None) -> tuple[GmailThread, ...]:
    return _GMAIL.validate_json(_read("gmail_threads.json", directory))


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def score(query_tokens: Sequence[str], title: str, body: str) -> int:
    """Keyword relevance: title hits count triple. Deterministic and dependency-free."""
    if not query_tokens:
        return 0
    title_toks, body_toks = tokens(title), tokens(body)
    return sum(3 * title_toks.count(t) + body_toks.count(t) for t in set(query_tokens))


def snippet(text: str, query_tokens: Sequence[str], width: int = 160) -> str:
    lowered = text.lower()
    hits = [i for t in query_tokens if (i := lowered.find(t)) >= 0]
    start = max(0, min(hits) - width // 4) if hits else 0
    out = text[start : start + width].replace("\n", " ").strip()
    return ("..." if start else "") + out + ("..." if start + width < len(text) else "")


def rank[T: BaseModel](
    items: Iterable[T],
    query: str,
    *,
    text_of: Callable[[T], tuple[str, str]],
    recency: Callable[[T], datetime],
    ident: Callable[[T], str],
) -> list[T]:
    """Score > 0 matches (or everything for an empty query), best score then newest first."""
    q = tokens(query)
    scored = [(score(q, *text_of(item)), item) for item in items]
    if q:
        scored = [(s, item) for s, item in scored if s > 0]
    scored.sort(key=lambda si: (-si[0], -recency(si[1]).timestamp(), ident(si[1])))
    return [item for _, item in scored]


def to_json(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def results_json(hits: Sequence[BaseModel], total: int) -> str:
    return to_json({"results": [h.model_dump(mode="json") for h in hits], "total": total})


EMPTY_RESULTS = to_json({"results": [], "total": 0})
