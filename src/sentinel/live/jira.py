"""Read-only Jira Cloud client (REST v3) mapped to Sentinel's Jira record shape.

Allowed requests: GET /rest/api/3/search/jql and GET /rest/api/3/issue/{key}. JQL is built
from typed filters with quoting; free-text goes through `text ~ "..."` only.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx

from sentinel.live.http import LiveAPIError, guarded_client, request_json

ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,9}-\d{1,7}$")
PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,9}$")
SEARCH_FIELDS = "summary,status,priority,assignee,updated,issuelinks"
ISSUE_FIELDS = (
    "summary,description,status,priority,assignee,reporter,labels,issuelinks,created,updated,"
    "comment,parent"
)
MAX_SEARCH_PAGES = 10


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_jql(
    query: str = "",
    status: str | None = None,
    assignee: str | None = None,
    label: str | None = None,
    blocked_by: str | None = None,
    project_keys: Sequence[str] = (),
) -> str:
    clauses: list[str] = []
    if project_keys:
        bad = [k for k in project_keys if not PROJECT_KEY.match(k)]
        if bad:
            raise LiveAPIError(f"invalid Jira project keys: {bad}")
        clauses.append(f"project in ({', '.join(project_keys)})")
    if query.strip():
        clauses.append(f"text ~ {quote(query.strip())}")
    if status:
        clauses.append(f"status = {quote(status)}")
    if assignee:
        clauses.append(f"assignee = {quote(assignee)}")
    if label:
        clauses.append(f"labels = {quote(label)}")
    if blocked_by:
        key = blocked_by.strip().upper()
        if not ISSUE_KEY.match(key):
            raise LiveAPIError(f"invalid Jira issue key: {blocked_by!r}")
        clauses.append(f'issue in linkedIssues("{key}", "is blocked by")')
    if not clauses:  # the /search/jql endpoint rejects unbounded queries
        clauses.append("updated >= -3650d")
    return " AND ".join(clauses) + " ORDER BY updated DESC"


def adf_text(node: Any) -> str:
    """Plain text from Atlassian Document Format (paragraph/heading/list per line)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_text(n) for n in node)
    kind = node.get("type")
    if kind == "text":
        return str(node.get("text", ""))
    if kind == "hardBreak":
        return "\n"
    inner = adf_text(node.get("content"))
    if kind in {"paragraph", "heading", "listItem", "codeBlock", "blockquote"}:
        return inner.strip() + "\n"
    return inner


def jira_time(value: str) -> str:
    """'2026-10-31T09:00:00.000+0000' -> ISO 8601 with a colon offset."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z").isoformat()


def _person(user: dict[str, Any] | None) -> str | None:
    if not user:
        return None
    return str(user.get("emailAddress") or user.get("displayName") or user.get("accountId"))


def blocked_by_keys(links: list[dict[str, Any]] | None) -> list[str]:
    keys = []
    for link in links or []:
        inward = (link.get("type") or {}).get("inward", "")
        if inward.lower() == "is blocked by" and "inwardIssue" in link:
            keys.append(str(link["inwardIssue"]["key"]))
    return keys


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        project_keys: Sequence[str] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url.startswith("https://"):
            raise LiveAPIError("JIRA_BASE_URL must be an https:// URL")
        self.base = base_url.rstrip("/")
        self.project_keys = tuple(project_keys)
        auth = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self.http = guarded_client(
            (
                ("GET", f"{self.base}/rest/api/3/search/jql"),
                ("GET", f"{self.base}/rest/api/3/issue/"),
            ),
            headers={"Authorization": f"Basic {auth}", "Accept": "application/json"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    def _url(self, key: str) -> str:
        return f"{self.base}/browse/{key}"

    async def search_issues(
        self,
        query: str,
        status: str | None,
        assignee: str | None,
        label: str | None,
        blocked_by: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        jql = build_jql(query, status, assignee, label, blocked_by, self.project_keys)
        wanted = offset + limit
        found: list[dict[str, Any]] = []
        token: str | None = None
        is_last = True
        for _ in range(MAX_SEARCH_PAGES):
            params: dict[str, Any] = {"jql": jql, "maxResults": 50, "fields": SEARCH_FIELDS}
            if token:
                params["nextPageToken"] = token
            data = await request_json(
                self.http, "GET", f"{self.base}/rest/api/3/search/jql", params=params
            )
            found += data.get("issues", [])
            token = data.get("nextPageToken")
            is_last = bool(data.get("isLast", not token))
            if is_last or not token or len(found) >= wanted:
                break
        hits = []
        for issue in found[offset:wanted]:
            fields = issue.get("fields") or {}
            hits.append(
                {
                    "key": issue["key"],
                    "url": self._url(issue["key"]),
                    "summary": fields.get("summary", ""),
                    "status": (fields.get("status") or {}).get("name", ""),
                    "priority": (fields.get("priority") or {}).get("name", ""),
                    "assignee": _person(fields.get("assignee")),
                    "blocked_by": blocked_by_keys(fields.get("issuelinks")),
                    "updated_at": jira_time(fields["updated"]),
                }
            )
        return hits, len(found) + (0 if is_last else 1)

    async def get_issue(self, key: str) -> dict[str, Any]:
        key = key.strip().upper()
        if not ISSUE_KEY.match(key):
            raise LiveAPIError(f"invalid Jira issue key: {key!r}")
        issue = await request_json(
            self.http,
            "GET",
            f"{self.base}/rest/api/3/issue/{key}",
            params={"fields": ISSUE_FIELDS},
        )
        fields = issue.get("fields") or {}
        comments = (fields.get("comment") or {}).get("comments", [])
        return {
            "key": issue["key"],
            "url": self._url(issue["key"]),
            "summary": fields.get("summary", ""),
            "description": adf_text(fields.get("description")).strip(),
            "status": (fields.get("status") or {}).get("name", ""),
            "priority": (fields.get("priority") or {}).get("name", ""),
            "assignee": _person(fields.get("assignee")),
            "reporter": _person(fields.get("reporter")) or "unknown",
            "labels": fields.get("labels") or [],
            "epic": (fields.get("parent") or {}).get("key"),
            "blocked_by": blocked_by_keys(fields.get("issuelinks")),
            "linked_pages": [],
            "created_at": jira_time(fields["created"]),
            "updated_at": jira_time(fields["updated"]),
            "comments": [
                {
                    "id": str(c.get("id")),
                    "author": _person(c.get("author")) or "unknown",
                    "created_at": jira_time(c["created"]),
                    "body": adf_text(c.get("body")).strip(),
                }
                for c in comments
            ],
        }
