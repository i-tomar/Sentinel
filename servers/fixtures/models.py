"""Record shapes served by the fixture MCP servers, plus the ground-truth manifest.

The manifest (`data/manifest.json`) is NOT served by any MCP tool: it is test/eval ground truth
(which records are poisoned, which docs are stale, which strings are seeded PII).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Source = Literal["notion", "jira", "gmail"]
IssueStatus = Literal["To Do", "In Progress", "Blocked", "In Review", "Done"]
Priority = Literal["Lowest", "Low", "Medium", "High", "Highest"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NotionPage(_Record):
    id: str
    title: str
    url: str
    owner: str
    created_at: datetime
    updated_at: datetime
    tags: tuple[str, ...]
    body: str


class JiraComment(_Record):
    id: str
    author: str
    created_at: datetime
    body: str


class JiraIssue(_Record):
    key: str
    url: str
    summary: str
    description: str
    status: IssueStatus
    priority: Priority
    assignee: str | None
    reporter: str
    labels: tuple[str, ...]
    epic: str | None
    blocked_by: tuple[str, ...]
    linked_pages: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    comments: tuple[JiraComment, ...]


class GmailMessage(_Record):
    id: str
    sender: str
    to: tuple[str, ...]
    cc: tuple[str, ...]
    sent_at: datetime
    body: str


class GmailThread(_Record):
    id: str
    url: str
    subject: str
    labels: tuple[str, ...]
    messages: tuple[GmailMessage, ...]


class PageHit(_Record):
    id: str
    title: str
    url: str
    owner: str
    updated_at: datetime
    snippet: str


class IssueHit(_Record):
    key: str
    url: str
    summary: str
    status: IssueStatus
    priority: Priority
    assignee: str | None
    blocked_by: tuple[str, ...]
    updated_at: datetime


class ThreadHit(_Record):
    id: str
    url: str
    subject: str
    participants: tuple[str, ...]
    labels: tuple[str, ...]
    message_count: int
    last_message_at: datetime
    snippet: str


class Draft(_Record):
    draft_id: str
    to: tuple[str, ...]
    subject: str
    body: str
    thread_id: str | None
    created_at: datetime


InjectionTechnique = Literal[
    "email_body",
    "page_footer",
    "ticket_comment",
    "html_comment",
    "base64_blob",
    "markdown_link_text",
    "system_prefix",
    "hindi",
    "hinglish",
    "tool_call_json",
    "delimiter_escape",
]


class PoisonedRecord(_Record):
    source: Source
    record_id: str
    technique: InjectionTechnique
    attacker_goal: str
    payload_marker: str
    """A substring of the record that is part of the injected instruction (or its encoding)."""
    forbidden_strings: tuple[str, ...]
    """Strings that must never appear in an answer or tool argument (e.g. attacker addresses)."""


class StalePair(_Record):
    source: Source
    stale_id: str
    current_id: str
    topic: str
    stale_fact: str
    current_fact: str


class BenignLookalike(_Record):
    source: Source
    record_id: str
    why_it_looks_suspicious: str


PiiKind = Literal["phone_in", "pan", "aadhaar", "email_personal"]


class PiiSeed(_Record):
    kind: PiiKind
    value: str
    source: Source
    record_id: str


class FixtureManifest(_Record):
    seed: int
    now: datetime
    counts: dict[Source, int]
    poisoned: tuple[PoisonedRecord, ...]
    stale_pairs: tuple[StalePair, ...]
    benign_lookalikes: tuple[BenignLookalike, ...]
    pii_seeds: tuple[PiiSeed, ...]


class FixtureDataset(_Record):
    notion: tuple[NotionPage, ...]
    jira: tuple[JiraIssue, ...]
    gmail: tuple[GmailThread, ...]
    manifest: FixtureManifest
