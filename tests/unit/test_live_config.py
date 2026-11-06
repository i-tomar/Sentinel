from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from sentinel.config import Settings
from sentinel.mcp_clients.pool import active_sources, live_env, live_server_params

pytestmark = pytest.mark.unit

FULL = {
    "data_mode": "live",
    "live_sources": "notion,jira,gmail",
    "notion_token": "ntn-secret",
    "jira_base_url": "https://acme.atlassian.test",
    "jira_email": "me@acme.test",
    "jira_api_token": "jira-secret",
    "jira_project_keys": "proj, ops",
    "gmail_client_id": "cid.apps.test",
    "gmail_client_secret": "gmail-secret",
    "gmail_refresh_token": "refresh-secret",
}


def test_defaults_keep_live_off_and_notion_first() -> None:
    s = Settings()
    assert s.data_mode == "fixture"
    assert s.live_sources == ("notion",)
    assert s.live_max_records == 200
    assert s.live_missing() == {"notion": ["NOTION_TOKEN"]}


def test_parsing_and_validation() -> None:
    s = Settings.model_validate(FULL)
    assert s.live_sources == ("notion", "jira", "gmail")
    assert s.jira_project_keys == ("PROJ", "OPS")
    assert s.live_missing() == {"notion": [], "jira": [], "gmail": []}
    with pytest.raises(ValidationError):
        Settings.model_validate({"live_sources": "notion,slack"})
    with pytest.raises(ValidationError, match="JIRA_PROJECT_KEYS"):
        Settings.model_validate({"jira_project_keys": "PROJ) OR (1=1"})


def test_live_missing_reports_each_source() -> None:
    s = Settings.model_validate(
        {"live_sources": "jira,gmail", "jira_base_url": "http://plain.test"}
    )
    assert s.live_missing() == {
        "jira": ["JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_BASE_URL (must start with https://)"],
        "gmail": ["GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"],
    }
    own_server = Settings.model_validate({"live_sources": "gmail", "gmail_mcp_command": "x"})
    assert own_server.live_missing() == {"gmail": []}


def test_each_live_server_only_gets_its_own_credentials() -> None:
    s = Settings.model_validate(FULL)
    assert live_env("notion", s) == {"NOTION_TOKEN": "ntn-secret"}
    assert live_env("jira", s) == {
        "JIRA_BASE_URL": "https://acme.atlassian.test",
        "JIRA_EMAIL": "me@acme.test",
        "JIRA_API_TOKEN": "jira-secret",
        "JIRA_PROJECT_KEYS": "PROJ,OPS",
    }
    assert set(live_env("gmail", s)) == {
        "GMAIL_CLIENT_ID",
        "GMAIL_CLIENT_SECRET",
        "GMAIL_REFRESH_TOKEN",
    }
    params = live_server_params("notion", s)
    assert (params.command, params.args) == (sys.executable, ["-m", "servers.live.notion_mcp"])


def test_gmail_command_override() -> None:
    s = Settings.model_validate({**FULL, "gmail_mcp_command": "node gmail-server.js --readonly"})
    params = live_server_params("gmail", s)
    assert (params.command, params.args) == ("node", ["gmail-server.js", "--readonly"])


def test_active_sources() -> None:
    assert active_sources(Settings(), ["notion", "jira", "gmail"]) == ["notion", "jira", "gmail"]
    live = Settings.model_validate({"data_mode": "live", "live_sources": "jira"})
    assert active_sources(live, ["notion", "jira", "gmail"]) == ["jira"]
