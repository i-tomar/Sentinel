"""HTTP 429 handling for rate-limited LLM APIs (Groq's free tier).

`RateLimitTransport` / `AsyncRateLimitTransport` wrap an httpx transport. On a 429 they wait
(the server's `retry-after` seconds when given, otherwise exponential backoff) and resend the
same request, up to `retries` times. Every wait is added to a shared `RateLimitWaits`, so the
eval harness can leave rate-limit waits out of agent latency.

If the server asks for a wait longer than `max_wait_s`, or the retries run out, the 429 is
returned unchanged (the client raises its own rate-limit error) and `exhausted` is set, so a
run can stop with a clear message instead of failing case after case.

Nothing here reads or logs request headers: the API key stays inside the request object.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio
import httpx

TOO_MANY_REQUESTS = 429


@dataclass
class RateLimitWaits:
    """Shared across calls: total seconds waited, retries made, and whether a 429 was given up."""

    waited_s: float = 0.0
    retries: int = 0
    exhausted: bool = False
    exhausted_reason: str = ""


@dataclass(frozen=True)
class RateLimitPolicy:
    retries: int = 6
    """Resends per request after a 429."""
    max_wait_s: float = 120.0
    """Longest single wait. A server asking for more (e.g. a daily token limit) is not waited on."""
    backoff_s: float = 2.0
    """First wait when the server sends no usable `retry-after`; doubles each retry."""


def retry_after_s(response: httpx.Response) -> float | None:
    """`retry-after` in seconds (the form Groq documents), or None if absent or not a number."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def next_wait(
    response: httpx.Response, attempt: int, policy: RateLimitPolicy, waits: RateLimitWaits
) -> float | None:
    """Seconds to wait before resending, or None to give up (marks `waits.exhausted`)."""
    if attempt >= policy.retries:
        waits.exhausted = True
        waits.exhausted_reason = f"still rate limited after {policy.retries} retries"
        return None
    asked = retry_after_s(response)
    delay = asked if asked is not None else policy.backoff_s * (2**attempt)
    if delay > policy.max_wait_s:
        waits.exhausted = True
        waits.exhausted_reason = (
            f"the server asked to wait {delay:.0f}s, more than the {policy.max_wait_s:.0f}s "
            "limit (possibly a daily limit)"
        )
        return None
    waits.waited_s += delay
    waits.retries += 1
    return delay


class RateLimitTransport(httpx.BaseTransport):
    def __init__(
        self,
        inner: httpx.BaseTransport,
        waits: RateLimitWaits,
        policy: RateLimitPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.waits = waits
        self.policy = policy or RateLimitPolicy()
        self.sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        attempt = 0
        while True:
            response = self.inner.handle_request(request)
            if response.status_code != TOO_MANY_REQUESTS:
                return response
            delay = next_wait(response, attempt, self.policy, self.waits)
            if delay is None:
                return response
            response.close()
            self.sleep(delay)
            attempt += 1

    def close(self) -> None:
        self.inner.close()


class AsyncRateLimitTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        waits: RateLimitWaits,
        policy: RateLimitPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self.inner = inner
        self.waits = waits
        self.policy = policy or RateLimitPolicy()
        self.sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        attempt = 0
        while True:
            response = await self.inner.handle_async_request(request)
            if response.status_code != TOO_MANY_REQUESTS:
                return response
            delay = next_wait(response, attempt, self.policy, self.waits)
            if delay is None:
                return response
            await response.aclose()
            await self.sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        await self.inner.aclose()
