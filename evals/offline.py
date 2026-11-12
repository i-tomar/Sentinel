"""Refuse public network connections during a mock eval run (CI's "no public network" rule).

The same socket-level backstop the test suite uses: any connection or DNS lookup to a
non-loopback host raises `NetworkBlocked`. Fixture MCP servers run in-process and Qdrant runs
in memory, so a mock run needs no network at all; a real-model run does not use this.
"""

from __future__ import annotations

import asyncio
import asyncio.proactor_events
import asyncio.selector_events
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkBlocked(RuntimeError):  # noqa: N818 - reads as a statement at call sites
    pass


def _check(host: Any) -> None:
    if host is not None and str(host) not in _LOOPBACK:
        raise NetworkBlocked(f"eval run tried to reach {host!r}; mock evals must stay offline")


@contextmanager
def block_network() -> Iterator[None]:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    loops = (
        asyncio.proactor_events.BaseProactorEventLoop,
        asyncio.selector_events.BaseSelectorEventLoop,
    )
    real_sock_connect = {cls: cls.sock_connect for cls in loops}

    def connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address:
            _check(address[0])
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if isinstance(address, tuple) and address:
            _check(address[0])
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded(cls: Any) -> Any:
        real = real_sock_connect[cls]

        async def sock_connect(self: Any, sock: socket.socket, address: Any) -> None:
            if isinstance(address, tuple) and address:
                _check(address[0])
            await real(self, sock, address)

        return sock_connect

    socket.socket.connect = connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = getaddrinfo
    for cls in loops:
        cls.sock_connect = guarded(cls)  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = real_getaddrinfo
        for cls in loops:
            cls.sock_connect = real_sock_connect[cls]  # type: ignore[method-assign]
