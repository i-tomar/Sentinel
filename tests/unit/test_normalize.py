from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from sentinel.retrieval.normalize import gmail_document, jira_document, notion_document
from servers.fixtures.common import load_gmail, load_jira, load_notion

pytestmark = pytest.mark.unit

FETCHED = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)


def _raw(model: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = model.model_dump(mode="json")
    return dumped


def test_notion_page() -> None:
    page = next(p for p in load_notion() if p.id == "auth-migration-v2")
    doc = notion_document(_raw(page), FETCHED)
    assert doc.key == "notion:auth-migration-v2"
    assert (doc.title, doc.url, doc.text) == (page.title, page.url, page.body)
    assert doc.updated_at == page.updated_at
    assert doc.fetched_at == FETCHED
    assert doc.metadata["owner"] == "priya.sharma@acme.test"


def test_jira_issue_includes_links_and_comments() -> None:
    issue = next(i for i in load_jira() if i.key == "PROJ-155")
    doc = jira_document(_raw(issue), FETCHED)
    assert doc.key == "jira:PROJ-155"
    assert doc.title.startswith("PROJ-155: ")
    assert "Status: To Do" in doc.text
    for comment in issue.comments:
        assert comment.body in doc.text  # untrusted comments are indexed; guards run later
    assert doc.metadata["linked_pages"] == "auth-migration-v2,customer-escalation-process"
    assert doc.updated_at == issue.updated_at


def test_jira_blocked_by_in_text_and_metadata() -> None:
    issue = next(i for i in load_jira() if i.key == "PROJ-142")
    doc = jira_document(_raw(issue), FETCHED)
    assert "Blocked by: PROJ-120." in doc.text
    assert doc.metadata["blocked_by"] == "PROJ-120"


def test_gmail_thread_uses_last_message_time() -> None:
    thread = next(t for t in load_gmail() if t.id == "thr-0003")
    doc = gmail_document(_raw(thread), FETCHED)
    assert doc.updated_at == thread.messages[-1].sent_at
    assert doc.author == "dana.whitfield@globex.test"
    assert doc.text.count("From: ") == len(thread.messages)
    assert doc.metadata["messages"] == "2"


def test_unknown_fields_are_ignored_but_missing_fields_fail() -> None:
    page = _raw(next(iter(load_notion())))
    assert notion_document({**page, "new_field": 1}, FETCHED).id == page["id"]
    del page["body"]
    with pytest.raises(ValidationError):
        notion_document(page, FETCHED)


def test_empty_thread_is_rejected() -> None:
    thread = _raw(next(iter(load_gmail())))
    with pytest.raises(ValueError, match="no messages"):
        gmail_document({**thread, "messages": []}, FETCHED)
