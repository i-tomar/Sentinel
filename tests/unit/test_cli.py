from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel import __version__, infra
from sentinel.cli import app
from sentinel.config import get_settings

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_infra_init_then_app_env(isolated_env: Path) -> None:
    result = runner.invoke(app, ["infra", "init"])
    assert result.exit_code == 0, result.output
    stored = infra.read_infra_env(isolated_env / ".env.infra")
    assert stored["LANGFUSE_INIT_USER_PASSWORD"] not in result.output

    again = runner.invoke(app, ["infra", "init"])
    assert again.exit_code == 1
    assert infra.read_infra_env(isolated_env / ".env.infra") == stored

    app_env = runner.invoke(app, ["infra", "app-env"])
    assert app_env.exit_code == 0
    assert app_env.stdout.splitlines() == infra.app_env_lines(stored)


def test_infra_app_env_without_init_fails() -> None:
    result = runner.invoke(app, ["infra", "app-env"])
    assert result.exit_code == 1
    assert "sentinel infra init" in result.output


def test_infra_check_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    result = runner.invoke(app, ["infra", "check", "--skip-langfuse"])
    assert result.exit_code == 1
    assert "FAIL qdrant" in result.stdout


def test_live_ingest_without_credentials_names_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_MODE", "live")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
    assert "notion: missing NOTION_TOKEN" in result.output
    monkeypatch.setenv("LIVE_SOURCES", "jira")
    get_settings.cache_clear()
    only_gmail = runner.invoke(app, ["ingest", "-s", "gmail"])
    assert only_gmail.exit_code == 2
    assert "none of the requested sources is in LIVE_SOURCES" in only_gmail.output


def test_live_check_lists_missing_vars_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_SOURCES", "notion,jira,gmail")
    monkeypatch.setenv("JIRA_BASE_URL", "http://insecure.test")
    result = runner.invoke(app, ["live", "check"])
    assert result.exit_code == 1
    out = result.stdout
    assert "notion  missing NOTION_TOKEN" in out
    assert "JIRA_BASE_URL (must start with https://)" in out
    assert "gmail   missing GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN" in out
    assert "https://www.googleapis.com/auth/gmail.readonly" in out
    monkeypatch.setenv("LIVE_SOURCES", "notion")
    monkeypatch.setenv("NOTION_TOKEN", "secret_value_123")
    get_settings.cache_clear()
    ready = runner.invoke(app, ["live", "check"])
    assert ready.exit_code == 0
    assert "notion  ready" in ready.stdout
    assert "secret_value_123" not in ready.stdout


def test_live_smoke_needs_live_mode_and_explicit_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.invoke(app, ["live", "smoke", "-s", "notion"]).exit_code == 2
    monkeypatch.setenv("DATA_MODE", "live")
    monkeypatch.setenv("NOTION_TOKEN", "secret_value_123")
    get_settings.cache_clear()
    dry = runner.invoke(app, ["live", "smoke", "-s", "notion"])  # no --yes: describes only
    assert dry.exit_code == 1
    assert "would call on notion: search_pages(query='', limit=1); get_page(<first result>)" in (
        dry.stdout
    )


def test_gmail_auth_needs_client_credentials() -> None:
    result = runner.invoke(app, ["live", "gmail-auth"])
    assert result.exit_code == 2
    assert "GMAIL_CLIENT_ID" in result.output


def test_search_rejects_unknown_source_and_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    assert runner.invoke(app, ["search", "x", "-s", "slack"]).exit_code != 0
    assert runner.invoke(app, ["search", "x", "--mode", "fuzzy"]).exit_code != 0
    empty = runner.invoke(app, ["search", "x"])
    assert empty.exit_code == 0
    assert "no results" in empty.stdout


def test_config_prints_json_with_masked_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_TOKEN", "secret_notion_token")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "secret_notion_token" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["data_mode"] == "fixture"
    assert payload["notion_token"] == "**********"
