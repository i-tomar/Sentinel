from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sentinel.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep tests hermetic: no real `.env`, no inherited Sentinel env vars, fresh settings cache."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
