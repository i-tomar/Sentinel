"""HTTP 429 handling for rate-limited LLM APIs: waits, retries, give-up rules (mocked HTTP)."""

from __future__ import annotations

import anyio
import httpx
import pytest

from sentinel.llm_ratelimit import (
    AsyncRateLimitTransport,
    RateLimitPolicy,
    RateLimitTransport,
    RateLimitWaits,
    retry_after_s,
)

pytestmark = pytest.mark.unit

URL = "https://api.groq.com/openai/v1/chat/completions"


def _scripted(*responses: httpx.Response) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    return httpx.MockTransport(handler), seen


def _limited(retry_after: str | None = None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, json={"error": {"message": "rate limited"}})


def _send(transport: httpx.BaseTransport) -> httpx.Response:
    with httpx.Client(transport=transport) as client:
        return client.post(URL, json={"model": "m"})


def test_a_429_is_waited_out_and_the_same_request_resent() -> None:
    inner, seen = _scripted(_limited("2"), _limited("0.5"), httpx.Response(200, json={}))
    waits = RateLimitWaits()
    slept: list[float] = []
    response = _send(RateLimitTransport(inner, waits, sleep=slept.append))
    assert response.status_code == 200
    assert slept == [2.0, 0.5]  # the server's retry-after, in seconds
    assert (waits.waited_s, waits.retries, waits.exhausted) == (2.5, 2, False)
    assert len(seen) == 3
    assert {r.content for r in seen} == {seen[0].content}  # same body every time


def test_without_retry_after_the_wait_doubles() -> None:
    inner, _ = _scripted(_limited(), _limited("soon"), _limited(), httpx.Response(200))
    waits = RateLimitWaits()
    slept: list[float] = []
    policy = RateLimitPolicy(backoff_s=1.0)
    assert _send(RateLimitTransport(inner, waits, policy, sleep=slept.append)).status_code == 200
    assert slept == [1.0, 2.0, 4.0]  # an unparseable retry-after counts as missing
    assert waits.waited_s == 7.0


def test_other_statuses_pass_through_untouched() -> None:
    for status in (200, 400, 401, 500, 503):
        inner, seen = _scripted(httpx.Response(status))
        waits = RateLimitWaits()
        slept: list[float] = []
        assert _send(RateLimitTransport(inner, waits, sleep=slept.append)).status_code == status
        assert (len(seen), slept, waits.retries) == (1, [], 0)


def test_giving_up_after_the_retries_returns_the_429_and_marks_it() -> None:
    inner, seen = _scripted(*[_limited("1") for _ in range(3)])
    waits = RateLimitWaits()
    slept: list[float] = []
    policy = RateLimitPolicy(retries=2)
    response = _send(RateLimitTransport(inner, waits, policy, sleep=slept.append))
    assert response.status_code == 429  # the client raises its own rate-limit error
    assert len(seen) == 3
    assert slept == [1.0, 1.0]
    assert waits.exhausted
    assert "after 2 retries" in waits.exhausted_reason


def test_a_wait_longer_than_allowed_is_not_waited() -> None:
    inner, seen = _scripted(_limited("3600"))
    waits = RateLimitWaits()
    slept: list[float] = []
    policy = RateLimitPolicy(max_wait_s=120)
    response = _send(RateLimitTransport(inner, waits, policy, sleep=slept.append))
    assert response.status_code == 429
    assert (len(seen), slept, waits.waited_s) == (1, [], 0.0)
    assert waits.exhausted
    assert "3600s" in waits.exhausted_reason
    assert "120s" in waits.exhausted_reason


def test_the_async_transport_behaves_the_same() -> None:
    inner, seen = _scripted(_limited("1.5"), httpx.Response(200, json={}))
    waits = RateLimitWaits()
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def send() -> int:
        transport = AsyncRateLimitTransport(inner, waits, sleep=fake_sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return (await client.post(URL, json={"model": "m"})).status_code

    assert anyio.run(send) == 200
    assert slept == [1.5]
    assert (waits.waited_s, waits.retries, len(seen)) == (1.5, 1, 2)


@pytest.mark.parametrize(
    ("header", "expected"), [("2", 2.0), ("0.25", 0.25), ("-1", None), ("Wed", None)]
)
def test_retry_after_parsing(header: str, expected: float | None) -> None:
    assert retry_after_s(httpx.Response(429, headers={"retry-after": header})) == expected
