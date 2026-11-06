"""Synthetic API payloads shaped like the documented Notion, Jira Cloud and Gmail responses.

Hand-written from the public API reference (not recorded from any real account); every
person, address and id is invented and uses reserved .test domains.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx

PAGE_V2 = "0b8c7e9a-1d2f-4a3b-9c8d-7e6f5a4b3c2d"
PAGE_V1 = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
THREAD_ID = "18c2f0a9b7d3e4f1"


def notion_page(page_id: str, title: str, edited: str, *, trashed: bool = False) -> dict[str, Any]:
    return {
        "object": "page",
        "id": page_id,
        "created_time": "2026-09-01T10:00:00.000Z",
        "last_edited_time": edited,
        "created_by": {"object": "user", "id": "user-7f3a"},
        "in_trash": trashed,
        "url": f"https://app.notion.com/p/{page_id.replace('-', '')}",
        "properties": {
            "Name": {
                "id": "title",
                "type": "title",
                "title": [{"type": "text", "plain_text": title}],
            },
        },
    }


def notion_blocks(*lines: tuple[str, str]) -> dict[str, Any]:
    return {
        "object": "list",
        "results": [
            {"object": "block", "type": kind, kind: {"rich_text": [{"plain_text": text}]}}
            for kind, text in lines
        ]
        + [{"object": "block", "type": "divider", "divider": {}}],
        "has_more": False,
        "next_cursor": None,
    }


def adf(*paragraphs: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": p}]} for p in paragraphs
        ],
    }


def jira_issue(
    key: str, summary: str, status: str, *, blocked_by: str | None = None
) -> dict[str, Any]:
    links = []
    if blocked_by:
        links.append(
            {
                "id": "10001",
                "type": {"name": "Blocks", "inward": "is blocked by", "outward": "blocks"},
                "inwardIssue": {"key": blocked_by},
            }
        )
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": adf(f"{summary}.", "Details for the team."),
            "status": {"name": status},
            "priority": {"name": "High"},
            "assignee": {
                "accountId": "acc-1",
                "displayName": "Arjun Mehta",
                "emailAddress": "arjun@acme.test",
            },
            "reporter": {"accountId": "acc-2", "displayName": "Priya Sharma"},
            "labels": ["auth"],
            "issuelinks": links,
            "created": "2026-09-20T09:00:00.000+0000",
            "updated": "2026-10-31T09:00:00.000+0000",
            "parent": {"key": "AUTH-1"},
            "comment": {
                "comments": [
                    {
                        "id": "c1",
                        "author": {"displayName": "Sofia Alvarez"},
                        "created": "2026-10-30T12:00:00.000+0530",
                        "body": adf("Customer is locked out."),
                    }
                ]
            },
        },
    }


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def gmail_thread() -> dict[str, Any]:
    return {
        "id": THREAD_ID,
        "messages": [
            {
                "id": "m1",
                "threadId": THREAD_ID,
                "labelIds": ["INBOX", "Label_7"],
                "internalDate": "1761730000000",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [
                        {"name": "From", "value": "Dana Whitfield <dana@globex.test>"},
                        {"name": "To", "value": "Sofia <sofia@acme.test>, ops@acme.test"},
                        {"name": "Subject", "value": "SSO login failing"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": b64("Login fails since Monday.\r\nPROJ-147?")},
                        },
                        {"mimeType": "text/html", "body": {"data": b64("<p>Login fails</p>")}},
                    ],
                },
            },
            {
                "id": "m2",
                "threadId": THREAD_ID,
                "labelIds": ["SENT"],
                "internalDate": "1761740000000",
                "payload": {
                    "mimeType": "text/html",
                    "headers": [
                        {"name": "From", "value": "sofia@acme.test"},
                        {"name": "To", "value": "dana@globex.test"},
                        {"name": "Cc", "value": "Kabir <kabir@acme.test>"},
                    ],
                    "body": {"data": b64("<div>Escalated to <b>Kabir</b> &amp; team.</div>")},
                },
            },
        ],
    }


Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Mock transport that records requests and answers from a route function."""

    def __init__(self, route: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._route = route
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._route(request)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) if r.content else None for r in self.requests]
