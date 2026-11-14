"""Retries for rate-limited LLM APIs: 429 waits, 5xx and connection-error backoff, give-up
rules, and token accounting (mocked HTTP only)."""

from __future__ import annotations

import anyio
import httpx
import pytest

from sentinel.llm_ratelimit import (
    AsyncRateLimitTransport,
    ProviderUsage,
    RateLimitPolicy,
    RateLimitTransport,
    retry_after_s,
)

pytestmark = pytest.mark.unit

URL = "https://api.groq.com/openai/v1/chat/completions"
Step = httpx.Response | Exception


def _scripted(*steps: Step) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A transport that returns (or raises) each step in turn."""
    seen: list[httpx.Request] = []
    queue = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return httpx.MockTransport(handler), seen


def _limited(retry_after: str | None = None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, json={"error": {"message": "rate limited"}})


def _ok(total_tokens: int | None = None) -> httpx.Response:
    body: dict[str, object] = {"choices": []}
    if total_tokens is not None:
        body["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": total_tokens}
    return httpx.Response(200, json=body)


def _send(transport: httpx.BaseTransport) -> httpx.Response:
    with httpx.Client(transport=transport) as client:
        return client.post(URL, json={"model": "m"})


def _sync(
    *steps: Step, policy: RateLimitPolicy | None = None
) -> tuple[httpx.Response | Exception, list[httpx.Request], ProviderUsage, list[float]]:
    inner, seen = _scripted(*steps)
    usage = ProviderUsage()
    slept: list[float] = []
    try:
        outcome: httpx.Response | Exception = _send(
            RateLimitTransport(inner, usage, policy, sleep=slept.append)
        )
    except httpx.TransportError as exc:
        outcome = exc
    return outcome, seen, usage, slept


# --- 429 -----------------------------------------------------------------------------------


def test_a_429_is_waited_out_and_the_same_request_resent() -> None:
    response, seen, usage, slept = _sync(_limited("2"), _limited("0.5"), _ok())
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert slept == [2.0, 0.5]  # the server's retry-after, in seconds
    assert (usage.waited_s, usage.rate_limit_retries, usage.exhausted) == (2.5, 2, False)
    assert len(seen) == 3
    assert {r.content for r in seen} == {seen[0].content}  # same body every time


def test_without_retry_after_the_wait_doubles() -> None:
    policy = RateLimitPolicy(backoff_s=1.0)
    response, _, usage, slept = _sync(
        _limited(), _limited("soon"), _limited(), _ok(), policy=policy
    )
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert slept == [1.0, 2.0, 4.0]  # an unparseable retry-after counts as missing
    assert usage.waited_s == 7.0


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 422])
def test_other_statuses_pass_through_untouched(status: int) -> None:
    response, seen, usage, slept = _sync(httpx.Response(status))
    assert isinstance(response, httpx.Response)
    assert response.status_code == status
    assert (len(seen), slept, usage.rate_limit_retries, usage.server_retries) == (1, [], 0, 0)


def test_giving_up_after_the_retries_returns_the_429_and_marks_it() -> None:
    policy = RateLimitPolicy(retries=2)
    response, seen, usage, slept = _sync(*[_limited("1") for _ in range(3)], policy=policy)
    assert isinstance(response, httpx.Response)
    assert response.status_code == 429  # the client raises its own rate-limit error
    assert len(seen) == 3
    assert slept == [1.0, 1.0]
    assert usage.exhausted
    assert "after 2 retries" in usage.exhausted_reason


def test_a_wait_longer_than_allowed_is_not_waited() -> None:
    response, seen, usage, slept = _sync(_limited("3600"), policy=RateLimitPolicy(max_wait_s=120))
    assert isinstance(response, httpx.Response)
    assert response.status_code == 429
    assert (len(seen), slept, usage.waited_s) == (1, [], 0.0)
    assert usage.exhausted
    assert "3600s" in usage.exhausted_reason
    assert "120s" in usage.exhausted_reason


# --- 5xx and connection errors ------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_5xx_is_retried_with_backoff(status: int) -> None:
    response, seen, usage, slept = _sync(httpx.Response(status), httpx.Response(status), _ok())
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert slept == [1.0, 2.0]
    assert (usage.server_retries, usage.rate_limit_retries, usage.waited_s) == (2, 0, 3.0)
    assert len(seen) == 3
    assert not usage.exhausted  # a server error never stops the suite on its own


def test_the_5xx_backoff_is_capped() -> None:
    policy = RateLimitPolicy(server_retries=6, server_backoff_s=4.0, server_max_backoff_s=10.0)
    steps: list[Step] = [httpx.Response(503) for _ in range(5)]
    _, _, usage, slept = _sync(*steps, _ok(), policy=policy)
    assert slept == [4.0, 8.0, 10.0, 10.0, 10.0]
    assert usage.waited_s == 42.0


def test_a_persistent_5xx_returns_the_last_response() -> None:
    policy = RateLimitPolicy(server_retries=2)
    response, seen, usage, slept = _sync(*[httpx.Response(502) for _ in range(3)], policy=policy)
    assert isinstance(response, httpx.Response)
    assert response.status_code == 502  # the client raises its own server error
    assert (len(seen), slept, usage.server_retries) == (3, [1.0, 2.0], 2)


def test_a_dropped_connection_is_retried() -> None:
    request = httpx.Request("POST", URL)
    response, seen, usage, slept = _sync(
        httpx.ConnectError("connection refused", request=request),
        httpx.ReadTimeout("read timed out", request=request),
        _ok(),
    )
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert slept == [1.0, 2.0]
    assert usage.server_retries == 2
    assert len(seen) == 3


def test_a_connection_that_never_works_raises_after_the_retries() -> None:
    request = httpx.Request("POST", URL)
    errors: list[Step] = [httpx.ConnectError("refused", request=request) for _ in range(3)]
    outcome, seen, usage, slept = _sync(*errors, policy=RateLimitPolicy(server_retries=2))
    assert isinstance(outcome, httpx.ConnectError)
    assert (len(seen), slept, usage.server_retries) == (3, [1.0, 2.0], 2)


def test_429_and_5xx_retries_are_counted_separately() -> None:
    _, _, usage, slept = _sync(_limited("0.5"), httpx.Response(500), _limited("0.5"), _ok())
    assert slept == [0.5, 1.0, 0.5]
    assert (usage.rate_limit_retries, usage.server_retries, usage.waited_s) == (2, 1, 2.0)


# --- token accounting ----------------------------------------------------------------------


def test_tokens_from_successful_completions_are_counted() -> None:
    inner, _ = _scripted(_ok(150), _limited("0"), _ok(40), httpx.Response(400), _ok())
    usage = ProviderUsage()
    with httpx.Client(transport=RateLimitTransport(inner, usage, sleep=lambda _s: None)) as c:
        for _ in range(4):
            c.post(URL, json={})
    assert usage.tokens == 190  # 150 + 40; errors and bodies without usage count nothing


def test_an_unreadable_body_counts_no_tokens() -> None:
    _, _, usage, _ = _sync(httpx.Response(200, content=b"not json"))
    assert usage.tokens == 0


# --- async ---------------------------------------------------------------------------------


def test_the_async_transport_behaves_the_same() -> None:
    request = httpx.Request("POST", URL)
    inner, seen = _scripted(
        _limited("1.5"),
        httpx.Response(503),
        httpx.ConnectError("refused", request=request),
        _ok(77),
    )
    usage = ProviderUsage()
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def send() -> int:
        transport = AsyncRateLimitTransport(inner, usage, sleep=fake_sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return (await client.post(URL, json={"model": "m"})).status_code

    assert anyio.run(send) == 200
    assert slept == [1.5, 1.0, 2.0]
    assert (usage.rate_limit_retries, usage.server_retries, usage.waited_s) == (1, 2, 4.5)
    assert (usage.tokens, len(seen)) == (77, 4)


@pytest.mark.parametrize(
    ("header", "expected"), [("2", 2.0), ("0.25", 0.25), ("-1", None), ("Wed", None)]
)
def test_retry_after_parsing(header: str, expected: float | None) -> None:
    assert retry_after_s(httpx.Response(429, headers={"retry-after": header})) == expected
