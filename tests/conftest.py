from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sentinel.config import Settings, get_settings

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def anyio_backend() -> str:
    """Async tests use anyio's pytest plugin (`@pytest.mark.anyio`), matching the MCP SDK."""
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep tests hermetic: no real `.env`, no inherited Sentinel env vars, fresh settings cache."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a non-loopback connection, unless marked `network`.

    HTTP mocking libraries can silently miss a client (e.g. an SDK switching HTTP stacks), so
    the socket layer is the backstop that keeps CI offline and credential-free."""
    if request.node.get_closest_marker("network"):
        return
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def check(address: Any) -> None:
        if isinstance(address, tuple) and address and str(address[0]) not in _LOOPBACK:
            raise RuntimeError(f"test tried to open a network connection to {address[0]!r}")

    def connect(self: socket.socket, address: Any) -> None:
        check(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
