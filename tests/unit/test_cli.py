from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel import __version__, infra
from sentinel.cli import app

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


def test_ingest_refuses_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_MODE", "live")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 2
    assert "step (i)" in result.output


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
