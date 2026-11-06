"""HTTP plumbing shared by the live clients.

`ReadOnlyGuard` wraps the transport and refuses any request that is not on an explicit
(method, URL prefix) allowlist, so even a bug cannot send a write to a real account.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

DEFAULT_TIMEOUT_S = 15.0
MAX_ERROR_CHARS = 200


class LiveAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ReadOnlyGuard(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, allowed: Sequence[tuple[str, str]]) -> None:
        self.inner = inner
        self.allowed = tuple(allowed)

    def permits(self, method: str, url: str) -> bool:
        return any(method == m and url.startswith(prefix) for m, prefix in self.allowed)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if not self.permits(request.method, url):
            raise LiveAPIError(f"blocked non-allowlisted request: {request.method} {url}")
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def guarded_client(
    allowed: Sequence[tuple[str, str]],
    headers: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> httpx.AsyncClient:
    inner = transport or httpx.AsyncHTTPTransport(retries=0)
    return httpx.AsyncClient(
        transport=ReadOnlyGuard(inner, allowed),
        headers=dict(headers or {}),
        timeout=timeout_s,
        follow_redirects=False,
    )


async def request_json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> Any:
    """Send a request and return JSON; HTTP errors become LiveAPIError without echoing
    request headers (credentials)."""
    try:
        response = await client.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        raise LiveAPIError(f"timeout calling {method} {url}") from exc
    except httpx.TransportError as exc:
        raise LiveAPIError(f"network error calling {method} {url}: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:MAX_ERROR_CHARS]
        hint = ""
        if response.status_code == 429:
            hint = f" (retry after {response.headers.get('Retry-After', '?')}s)"
        raise LiveAPIError(
            f"{response.status_code} from {method} {url}{hint}: {detail}", response.status_code
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LiveAPIError(f"non-JSON response from {method} {url}") from exc
