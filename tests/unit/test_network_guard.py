from __future__ import annotations

import socket

import anyio
import httpx2
import pytest

pytestmark = pytest.mark.unit

# Never-routable destinations: TEST-NET-3 and the reserved .invalid TLD.
UNROUTABLE_IP = "203.0.113.10"
UNRESOLVABLE_HOST = "sentinel-guard.invalid"


def test_non_loopback_connections_are_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s,
        pytest.raises(RuntimeError, match="network connection"),
    ):
        s.connect((UNROUTABLE_IP, 443))


def test_name_resolution_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="network connection"):
        socket.getaddrinfo(UNRESOLVABLE_HOST, 443)


@pytest.mark.anyio
async def test_async_connections_are_blocked() -> None:
    with pytest.raises(BaseExceptionGroup) as info:  # anyio groups per-address attempts
        await anyio.connect_tcp(UNROUTABLE_IP, 443)
    assert info.group_contains(RuntimeError, match="network connection")


@pytest.mark.anyio
async def test_async_http_client_is_blocked() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(Exception, match="network connection"):
            await client.get(f"https://{UNRESOLVABLE_HOST}/")


def test_loopback_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
