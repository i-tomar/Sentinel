from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sentinel import __version__
from sentinel.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_config_prints_json_with_masked_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_TOKEN", "secret_notion_token")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "secret_notion_token" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["data_mode"] == "fixture"
    assert payload["notion_token"] == "**********"
