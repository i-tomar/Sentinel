from __future__ import annotations

import socket

import pytest

pytestmark = pytest.mark.unit


def test_non_loopback_connections_are_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s,
        pytest.raises(RuntimeError, match="network connection"),
    ):
        s.connect(("203.0.113.10", 443))  # TEST-NET-3: never routable, never contacted


def test_loopback_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
