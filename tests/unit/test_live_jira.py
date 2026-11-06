"""Live API clients against mock transports (no network, no real accounts)."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from sentinel.live.http import LiveAPIError
from sentinel.live.jira import JiraClient, adf_text, build_jql
from sentinel.retrieval.normalize import jira_document
from tests.live_payloads import (
    Recorder,
    jira_issue,
)

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

NOW = datetime(2026, 11, 2, tzinfo=UTC)


# --- Jira ---------------------------------------------------------------------------------


def test_jql_quotes_values_and_validates_keys() -> None:
    jql = build_jql('auth "migration"', status="Blocked", label="auth", project_keys=["PROJ"])
    assert jql == (
        'project in (PROJ) AND text ~ "auth \\"migration\\"" AND status = "Blocked" '
        'AND labels = "auth" ORDER BY updated DESC'
    )
    assert build_jql(blocked_by="proj-120") == (
        'issue in linkedIssues("PROJ-120", "is blocked by") ORDER BY updated DESC'
    )
    assert build_jql() == "updated >= -3650d ORDER BY updated DESC"  # bounded
    with pytest.raises(LiveAPIError):
        build_jql(blocked_by='PROJ-1") OR project = SECRET OR ("')
    with pytest.raises(LiveAPIError):
        build_jql(project_keys=["PROJ) OR (1=1"])


def test_adf_text() -> None:
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Line one"},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "two"},
                ],
            },
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "item"}]}
                        ],
                    }
                ],
            },
        ],
    }
    assert adf_text(doc).strip() == "Line one\ntwo\nitem"


def _jira_route(request: httpx.Request) -> httpx.Response:
    url = urlparse(str(request.url))
    if url.path == "/rest/api/3/search/jql":
        params = parse_qs(url.query)
        if params.get("nextPageToken") == ["t2"]:
            return httpx.Response(
                200,
                json={
                    "issues": [
                        jira_issue("PROJ-151", "Admin console", "Blocked", blocked_by="PROJ-120")
                    ],
                    "isLast": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "issues": [
                    jira_issue("PROJ-142", "Billing login", "Blocked", blocked_by="PROJ-120")
                ],
                "nextPageToken": "t2",
                "isLast": False,
            },
        )
    if url.path == "/rest/api/3/issue/PROJ-142":
        return httpx.Response(
            200, json=jira_issue("PROJ-142", "Billing login", "Blocked", blocked_by="PROJ-120")
        )
    return httpx.Response(401, json={"errorMessages": ["unauthorized"]})


async def test_jira_search_paginates_with_tokens() -> None:
    rec = Recorder(_jira_route)
    client = JiraClient(
        "https://acme.atlassian.test",
        "me@acme.test",
        "jira-secret",
        ["PROJ"],
        transport=rec.transport,
    )
    hits, total = await client.search_issues("", None, None, None, "PROJ-120", limit=5, offset=0)
    assert [h["key"] for h in hits] == ["PROJ-142", "PROJ-151"]
    assert hits[0]["blocked_by"] == ["PROJ-120"]
    assert total == 2
    jql = parse_qs(urlparse(str(rec.requests[0].url)).query)["jql"][0]
    assert jql.startswith("project in (PROJ) AND issue in linkedIssues")
    auth = rec.requests[0].headers["Authorization"]
    assert base64.b64decode(auth.split()[1]).decode() == "me@acme.test:jira-secret"


async def test_jira_issue_maps_to_record_shape() -> None:
    client = JiraClient(
        "https://acme.atlassian.test", "e", "t", transport=Recorder(_jira_route).transport
    )
    raw = await client.get_issue("proj-142")
    doc = jira_document(raw, NOW)
    assert doc.key == "jira:PROJ-142"
    assert doc.url == "https://acme.atlassian.test/browse/PROJ-142"
    assert "Blocked by: PROJ-120." in doc.text
    assert "Customer is locked out." in doc.text
    assert raw["comments"][0]["created_at"] == "2026-10-30T12:00:00+05:30"
    assert raw["assignee"] == "arjun@acme.test"


async def test_jira_rejects_plain_http_and_bad_keys() -> None:
    with pytest.raises(LiveAPIError, match="https"):
        JiraClient("http://acme.atlassian.test", "e", "t")
    client = JiraClient(
        "https://acme.atlassian.test", "e", "t", transport=Recorder(_jira_route).transport
    )
    with pytest.raises(LiveAPIError, match="invalid Jira issue key"):
        await client.get_issue("PROJ-1/../../myself")
    with pytest.raises(LiveAPIError, match="401"):
        await client.get_issue("PROJ-9")
