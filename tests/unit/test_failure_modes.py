from __future__ import annotations

import json
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from servers.fixtures import failure_modes as fm

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


def test_parse_spec() -> None:
    assert fm.parse_spec("") == {}
    assert fm.parse_spec(" gmail=timeout , jira.get_issue=error_500,*=empty ,") == {
        "gmail": "timeout",
        "jira.get_issue": "error_500",
        "*": "empty",
    }


@pytest.mark.parametrize("spec", ["gmail", "=timeout", "gmail=explode", "gmail=timeout,jira"])
def test_parse_spec_rejects_bad_rules(spec: str) -> None:
    with pytest.raises(ValueError, match="failure"):
        fm.parse_spec(spec)


def test_most_specific_rule_wins() -> None:
    env = {fm.FAILURES_ENV: "*=rate_limit,gmail=empty,gmail.get_thread=malformed"}
    assert fm.mode_for("gmail", "get_thread", env) == "malformed"
    assert fm.mode_for("gmail", "search_threads", env) == "empty"
    assert fm.mode_for("jira", "get_issue", env) == "rate_limit"
    assert fm.mode_for("jira", "get_issue", {}) is None


async def test_apply_without_rules_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fm.FAILURES_ENV, raising=False)
    assert await fm.apply("notion", "get_page", empty="[]") is None


@pytest.mark.parametrize(
    ("mode", "message"),
    [("error_500", "500 Internal Server Error"), ("rate_limit", "429 Too Many Requests")],
)
async def test_apply_error_modes(monkeypatch: pytest.MonkeyPatch, mode: str, message: str) -> None:
    monkeypatch.setenv(fm.FAILURES_ENV, f"jira={mode}")
    with pytest.raises(ToolError, match=message):
        await fm.apply("jira", "search_issues", empty="[]")


async def test_apply_empty_and_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fm.FAILURES_ENV, "notion=empty,gmail=malformed")
    assert await fm.apply("notion", "search_pages", empty='{"results": []}') == '{"results": []}'
    malformed = await fm.apply("gmail", "get_thread", empty="{}")
    assert malformed is not None
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed)


async def test_apply_timeout_stalls_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fm.FAILURES_ENV, "gmail=timeout")
    monkeypatch.setenv(fm.TIMEOUT_ENV, "0.2")
    start = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        await fm.apply("gmail", "search_threads", empty="[]")
    assert time.monotonic() - start >= 0.19
