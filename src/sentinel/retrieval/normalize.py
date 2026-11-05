"""Turn MCP tool payloads (fixture record shapes) into `Document`s.

The wire models ignore unknown fields so a server can add data without breaking ingestion.
Live connectors (step i) will add their own normalizers that produce the same `Document`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from sentinel.retrieval.models import Document


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
