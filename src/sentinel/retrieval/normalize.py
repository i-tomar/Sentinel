"""Turn MCP tool payloads (fixture record shapes) into `Document`s.

The wire models ignore unknown fields so a server can add data without breaking ingestion.
Live connectors (step i) will add their own normalizers that produce the same `Document`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from sentinel.retrieval.models import Document

FIELD_LABELS = (
    "Status",
    "Priority",
    "Assignee",
    "Reporter",
    "Blocked by",
    "Linked pages",
    "Comments",
    "From",
    "To",
    "Cc",
    "Date",
)
_FIELD_LABEL = re.compile(r"\b(?:" + "|".join(map(re.escape, FIELD_LABELS)) + r"):")


def strip_field_labels(text: str) -> str:
    """Drop the field labels the normalizers add ("Status:", "From:"...). Every record has
    them, so they say nothing about whether a record matches a query."""
    return _FIELD_LABEL.sub(" ", text)


_GMAIL_HEADER = re.compile(r"^(?P<label>From|To|Cc|Date): (?P<value>[^\n]*)$", re.M)
_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def mask_generated_fields(text: str, source: str) -> str:
    """Blank (same length, so offsets stay valid) what `gmail_document` itself generates on
    its header lines: the fixed labels "From: ", "To: ", "Cc: ", "Date: ", each value that is
    a complete email address, and a Date value that is an ISO timestamp.

    For the injection scan only. Everything sender-controlled stays visible: display names,
    raw headers that did not parse as an address, odd recipient strings, the subject (the
    title) and the body. A body line that exactly imitates a header line, i.e. a label
    followed only by valid addresses, is blanked too; it contains no words to scan.
    """
    if source != "gmail":
        return text
    chars = list(text)

    def blank(start: int, end: int) -> None:
        chars[start:end] = " " * (end - start)

    for m in _GMAIL_HEADER.finditer(text):
        start, value = m.start("value"), m.group("value")
        blank(m.start("label"), start)
        if m.group("label") == "Date":
            try:
                datetime.fromisoformat(value)
            except ValueError:
                continue
            blank(start, m.end("value"))
            continue
        offset = start
        for part in value.split(", "):
            if _ADDRESS.fullmatch(part):
                blank(offset, offset + len(part))
            offset += len(part) + 2
    return "".join(chars)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NotionPageIn(_Wire):
    id: str
    title: str
    url: str
    owner: str
    updated_at: datetime
    tags: list[str] = []
    body: str


class JiraCommentIn(_Wire):
    author: str
    created_at: datetime
    body: str


class JiraIssueIn(_Wire):
    key: str
    url: str
    summary: str
    description: str
    status: str
    priority: str
    assignee: str | None
    reporter: str
    labels: list[str] = []
    epic: str | None = None
    blocked_by: list[str] = []
    linked_pages: list[str] = []
    updated_at: datetime
    comments: list[JiraCommentIn] = []


class GmailMessageIn(_Wire):
    sender: str
    to: list[str]
    cc: list[str] = []
    sent_at: datetime
    body: str


class GmailThreadIn(_Wire):
    id: str
    url: str
    subject: str
    labels: list[str] = []
    messages: list[GmailMessageIn]


def notion_document(raw: Any, fetched_at: datetime) -> Document:
    page = NotionPageIn.model_validate(raw)
    return Document(
        source="notion",
        id=page.id,
        url=page.url,
        title=page.title,
        text=page.body,
        updated_at=page.updated_at,
        fetched_at=fetched_at,
        author=page.owner,
        metadata={"owner": page.owner, "tags": ",".join(page.tags)},
    )


def jira_document(raw: Any, fetched_at: datetime) -> Document:
    issue = JiraIssueIn.model_validate(raw)
    header = [
        f"{issue.key}: {issue.summary}",
        f"Status: {issue.status}. Priority: {issue.priority}.",
        f"Assignee: {issue.assignee or 'unassigned'}. Reporter: {issue.reporter}.",
    ]
    if issue.blocked_by:
        header.append(f"Blocked by: {', '.join(issue.blocked_by)}.")
    if issue.linked_pages:
        header.append(f"Linked pages: {', '.join(issue.linked_pages)}.")
    parts = ["\n".join(header), issue.description]
    if issue.comments:
        parts.append(
            "Comments:\n"
            + "\n".join(
                f"- {c.author} ({c.created_at.date().isoformat()}): {c.body}"
                for c in issue.comments
            )
        )
    return Document(
        source="jira",
        id=issue.key,
        url=issue.url,
        title=f"{issue.key}: {issue.summary}",
        text="\n\n".join(parts),
        updated_at=issue.updated_at,
        fetched_at=fetched_at,
        author=issue.reporter,
        metadata={
            "status": issue.status,
            "priority": issue.priority,
            "assignee": issue.assignee or "",
            "blocked_by": ",".join(issue.blocked_by),
            "linked_pages": ",".join(issue.linked_pages),
            "labels": ",".join(issue.labels),
        },
    )


def gmail_document(raw: Any, fetched_at: datetime) -> Document:
    thread = GmailThreadIn.model_validate(raw)
    if not thread.messages:
        raise ValueError(f"thread {thread.id} has no messages")
    messages = sorted(thread.messages, key=lambda m: m.sent_at)
    text = "\n\n".join(
        "\n".join(
            [
                f"From: {m.sender}",
                f"To: {', '.join(m.to)}",
                *([f"Cc: {', '.join(m.cc)}"] if m.cc else []),
                f"Date: {m.sent_at.isoformat()}",
                "",
                m.body,
            ]
        )
        for m in messages
    )
    return Document(
        source="gmail",
        id=thread.id,
        url=thread.url,
        title=thread.subject,
        text=text,
        updated_at=messages[-1].sent_at,
        fetched_at=fetched_at,
        author=messages[0].sender,
        metadata={"labels": ",".join(thread.labels), "messages": str(len(messages))},
    )
