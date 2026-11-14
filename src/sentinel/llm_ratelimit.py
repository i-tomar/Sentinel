"""Retries and usage accounting for rate-limited LLM APIs (Groq's free tier).

`RateLimitTransport` / `AsyncRateLimitTransport` wrap an httpx transport:

- **HTTP 429:** wait the server's `retry-after` seconds (exponential backoff when absent) and
  resend, up to `retries` times. A wait longer than `max_wait_s`, or running out of retries,
  returns the 429 unchanged (the client raises its own rate-limit error) and marks the usage
  `exhausted`, so a run can stop with a clear message instead of failing case after case.
- **HTTP 5xx and connection errors:** resend after a capped exponential backoff, up to
  `server_retries` times, then return the last response or raise the last error.
- **Usage:** every wait is added to `ProviderUsage.waited_s` (the eval harness keeps it out of
  latency), and the `usage.total_tokens` of every successful completion to `tokens` (the token
  budget guard reads it).

Nothing here reads or logs request headers: the API key stays inside the request object.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio
import httpx

TOO_MANY_REQUESTS = 429
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


@dataclass
class ProviderUsage:
    """Shared across calls: waits, retries, tokens, and whether a 429 was given up on."""

    waited_s: float = 0.0
    """All retry waits (429 and 5xx/connection backoff); never agent latency."""
    rate_limit_retries: int = 0
    server_retries: int = 0
    tokens: int = 0
    """Sum of `usage.total_tokens` over successful completions."""
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
    server_retries: int = 3
    """Resends per request after a 5xx or a connection error."""
    server_backoff_s: float = 1.0
    """First wait after a 5xx or connection error; doubles each retry, capped below."""
    server_max_backoff_s: float = 30.0


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
    response: httpx.Response, attempt: int, policy: RateLimitPolicy, usage: ProviderUsage
) -> float | None:
    """Seconds to wait before resending after a 429, or None to give up (marks `exhausted`)."""
    if attempt >= policy.retries:
        usage.exhausted = True
        usage.exhausted_reason = f"still rate limited after {policy.retries} retries"
        return None
    asked = retry_after_s(response)
    delay = asked if asked is not None else policy.backoff_s * (2**attempt)
    if delay > policy.max_wait_s:
        usage.exhausted = True
        usage.exhausted_reason = (
            f"the server asked to wait {delay:.0f}s, more than the {policy.max_wait_s:.0f}s "
            "limit (possibly a daily limit)"
        )
        return None
    usage.waited_s += delay
    usage.rate_limit_retries += 1
    return delay


def server_wait(attempt: int, policy: RateLimitPolicy, usage: ProviderUsage) -> float | None:
    """Seconds to wait before resending after a 5xx or connection error, or None to give up."""
    if attempt >= policy.server_retries:
        return None
    delay = min(policy.server_backoff_s * float(2**attempt), policy.server_max_backoff_s)
    usage.waited_s += delay
    usage.server_retries += 1
    return delay


def count_tokens(response: httpx.Response, usage: ProviderUsage) -> None:
    """Add a successful completion's `usage.total_tokens` (missing or odd bodies count 0)."""
    try:
        body = json.loads(response.content)
    except (ValueError, UnicodeDecodeError):
        return
    total = body.get("usage", {}).get("total_tokens") if isinstance(body, dict) else None
    if isinstance(total, int) and total > 0:
        usage.tokens += total


class RateLimitTransport(httpx.BaseTransport):
    def __init__(
        self,
        inner: httpx.BaseTransport,
        usage: ProviderUsage,
        policy: RateLimitPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.usage = usage
        self.policy = policy or RateLimitPolicy()
        self.sleep = sleep

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        limited = failed = 0
        while True:
            try:
                response = self.inner.handle_request(request)
            except httpx.TransportError:
                delay = server_wait(failed, self.policy, self.usage)
                if delay is None:
                    raise
                failed += 1
                self.sleep(delay)
                continue
            if response.status_code == TOO_MANY_REQUESTS:
                delay = next_wait(response, limited, self.policy, self.usage)
                limited += 1
            elif response.status_code in RETRYABLE_STATUS:
                delay = server_wait(failed, self.policy, self.usage)
                failed += 1
            else:
                if response.is_success:
                    response.read()
                    count_tokens(response, self.usage)
                return response
            if delay is None:
                return response
            response.close()
            self.sleep(delay)

    def close(self) -> None:
        self.inner.close()


class AsyncRateLimitTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        usage: ProviderUsage,
        policy: RateLimitPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    ) -> None:
        self.inner = inner
        self.usage = usage
        self.policy = policy or RateLimitPolicy()
        self.sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        limited = failed = 0
        while True:
            try:
                response = await self.inner.handle_async_request(request)
            except httpx.TransportError:
                delay = server_wait(failed, self.policy, self.usage)
                if delay is None:
                    raise
                failed += 1
                await self.sleep(delay)
                continue
            if response.status_code == TOO_MANY_REQUESTS:
                delay = next_wait(response, limited, self.policy, self.usage)
                limited += 1
            elif response.status_code in RETRYABLE_STATUS:
                delay = server_wait(failed, self.policy, self.usage)
                failed += 1
            else:
                if response.is_success:
                    await response.aread()
                    count_tokens(response, self.usage)
                return response
            if delay is None:
                return response
            await response.aclose()
            await self.sleep(delay)

    async def aclose(self) -> None:
        await self.inner.aclose()
