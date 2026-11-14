"""The eval harness with LLM_PROVIDER=groq on mocked HTTP (httpx.MockTransport): token totals,
rate-limit waits kept out of latency, pacing between cases, stopping on an exhausted limit,
and the API key never reaching a result or report. No request leaves the process."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from evals.datasets import load_cases
from evals.gate import evaluate_gate
from evals.harness import (
    HarnessConfig,
    RateLimitExhaustedError,
    TokenBudgetError,
    agent_latency_ms,
    over_budget,
    run_case,
    run_suite,
)
from evals.metrics import compute_metrics
from evals.report import build_report, case_failures, render_markdown
from evals.run import _progress_line
from evals.schema import EvalCase
from sentinel.config import Settings
from sentinel.graph.nodes.planner import make_planner
from sentinel.llm import make_chat_model, rate_limited_clients
from sentinel.llm_ratelimit import ProviderUsage, RateLimitPolicy

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

FAKE_KEY = "gsk_test_" + "s3cr3t" * 6
MODEL = "openai/gpt-oss-120b"
CASES = {c.id: c for c in load_cases()}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _completion(prompt: int, out: int) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "PROJ-142 is blocked by PROJ-120 [1]."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out},
    }


class FakeGroq:
    """Answers every chat completion. The first `limited` requests get a 429, or the status in
    `fail_status` with `fail_body` instead when that is set."""

    def __init__(
        self,
        limited: int = 0,
        retry_after: str = "0",
        fail_status: int = 429,
        fail_body: dict[str, Any] | None = None,
    ) -> None:
        self.limited = limited
        self.retry_after = retry_after
        self.fail_status = fail_status
        self.fail_body = fail_body or {"error": {"message": "slow down"}}
        self.requests: list[httpx.Request] = []
        self.answered = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "api.groq.com"
        if len(self.requests) <= self.limited:
            headers = {"retry-after": self.retry_after} if self.fail_status == 429 else {}
            return httpx.Response(self.fail_status, headers=headers, json=self.fail_body)
        self.answered += 1
        return httpx.Response(200, json=_completion(prompt=100 + self.answered, out=10))


def _config(fake: FakeGroq, policy: RateLimitPolicy | None = None, **changes: Any) -> HarnessConfig:
    settings = Settings(
        _env_file=None, llm_provider="groq", llm_model=MODEL, llm_api_key=SecretStr(FAKE_KEY)
    )
    waits = ProviderUsage()
    mock = httpx.MockTransport(fake)
    policy = policy or RateLimitPolicy()

    def factory() -> Any:
        clients = rate_limited_clients(waits, policy, sync_inner=mock, async_inner=mock)
        return make_chat_model(settings, clients, usage=waits)

    return HarnessConfig(
        llm_factory=factory,
        planner_factory=lambda model: make_planner("groq", model),
        tokens_reported=True,
        model_name=f"groq:{MODEL}",
        usage=waits,
        secrets=(FAKE_KEY,),  # as evals.run passes the configured key
        **changes,
    )


def _case(case_id: str) -> EvalCase:
    return CASES[case_id]


async def test_tokens_are_recorded_from_groq_responses() -> None:
    fake = FakeGroq()
    result = await run_case(_case("norm-001"), _config(fake))
    assert result.harness_error is None
    assert fake.answered >= 2  # the LLM planner and the answer
    assert result.input_tokens == sum(100 + i for i in range(1, fake.answered + 1))
    assert result.output_tokens == 10 * fake.answered
    metrics = compute_metrics([result], corpus_keys=set(), k=5)
    assert metrics.tokens is not None  # no longer None with a real provider
    assert metrics.flat()["tokens_mean_total"] == result.input_tokens + result.output_tokens


def test_latency_arithmetic() -> None:
    assert agent_latency_ms(3_400, 2.5) == (900, 2_500)
    assert agent_latency_ms(800, 0.0) == (800, 0)
    assert agent_latency_ms(100, 0.5) == (0, 500)  # never negative


async def test_rate_limit_waits_are_not_agent_latency() -> None:
    fake = FakeGroq(limited=1, retry_after="1")
    started = time.perf_counter()
    result = await run_case(_case("norm-001"), _config(fake))
    wall_ms = (time.perf_counter() - started) * 1000
    assert result.harness_error is None
    assert result.provider_wait_ms == 1000
    assert wall_ms >= 1000  # the wait really happened
    assert result.latency_ms <= wall_ms - 1000  # ...and is not counted as latency


async def test_cases_are_paced_and_the_pause_is_not_latency() -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    config = _config(FakeGroq(), case_delay_s=7.5, case_sleep=fake_sleep)
    cases = [_case("norm-001"), _case("norm-002"), _case("bl-d01")]
    results = await run_suite(cases, config)
    assert [r.harness_error for r in results] == [None, None, None]
    assert slept == [7.5, 7.5]  # between cases, not before the first
    assert all(r.provider_wait_ms == 0 for r in results)


async def test_an_exhausted_rate_limit_stops_the_suite() -> None:
    fake = FakeGroq(limited=1_000, retry_after="3600")  # e.g. a daily limit
    seen: list[str] = []
    config = _config(fake, RateLimitPolicy(max_wait_s=120))
    with pytest.raises(RateLimitExhaustedError, match=r"stopped after 1/3 cases.*3600s"):
        await run_suite(
            [_case("norm-001"), _case("norm-002"), _case("bl-d01")],
            config,
            lambda _i, _n, r: seen.append(r.case_id),
        )
    assert seen == ["norm-001"]
    assert len(fake.requests) == 1  # no wait was attempted, nothing was hammered


async def test_the_key_never_reaches_results_or_reports() -> None:
    fake = FakeGroq(limited=1, retry_after="0")
    results = [await run_case(_case(cid), _config(fake)) for cid in ("norm-001", "pii-001")]
    assert all(r.harness_error is None for r in results)
    assert all(r.headers["authorization"] == f"Bearer {FAKE_KEY}" for r in fake.requests)
    for result in results:
        assert FAKE_KEY not in result.model_dump_json()
    metrics = compute_metrics(results, corpus_keys=set(), k=5)
    gate = evaluate_gate(metrics.flat(), _missing_baseline())
    meta = {
        "model": f"groq:{MODEL}",
        "dataset_version": "1.0.0",
        "cases": len(results),
        "started_at": "2026-11-13T00:00:00+00:00",
        "rate_limit": {"retries": 1, "waited_s": 0.0},
    }
    report = build_report(results, corpus_keys=set(), gate=gate, meta=meta, k=5)
    markdown = render_markdown(report, gate)
    report.pop("_metrics_objects")
    assert FAKE_KEY not in json.dumps(report)
    assert FAKE_KEY not in markdown


def _missing_baseline() -> Path:
    return Path("does-not-exist-baseline.json")


async def test_server_error_waits_are_not_agent_latency() -> None:
    fake = FakeGroq(limited=1, fail_status=503, fail_body={"error": {"message": "overloaded"}})
    config = _config(fake)  # default server backoff: 1 s before the first resend
    started = time.perf_counter()
    result = await run_case(_case("norm-001"), config)
    wall_ms = (time.perf_counter() - started) * 1000
    assert result.harness_error is None
    assert config.usage is not None
    assert config.usage.server_retries == 1
    assert result.provider_wait_ms == 1000
    assert result.latency_ms <= wall_ms - 1000


async def test_a_failed_case_records_a_safe_reason() -> None:
    body = {
        "error": {
            "message": f"bad request from priya.sharma@acme.test with key {FAKE_KEY}",
            "type": "invalid_request_error",
        }
    }
    fake = FakeGroq(limited=1_000, fail_status=400, fail_body=body)
    result = await run_case(_case("norm-001"), _config(fake))
    # the real error, not the ExceptionGroup the agent's task groups wrap it in
    assert result.harness_error == "OpenAIInvalidRequestError"  # langchain's 400 subclass
    detail = result.harness_error_detail
    assert detail is not None
    assert detail.startswith("OpenAIInvalidRequestError: Error code: 400")
    assert "bad request from <EMAIL> with key <SECRET>" in detail
    assert FAKE_KEY not in detail
    assert "priya.sharma@acme.test" not in detail
    assert "\n" not in detail
    line = _progress_line(7, 52, result)
    assert line.startswith("[  7/52] norm-001   error: OpenAIInvalidRequestError: Error code: 400")
    assert FAKE_KEY not in line
    failures = case_failures(result)
    assert failures == [f"harness error ({detail})"]


async def test_the_token_budget_stops_before_crossing_it() -> None:
    fake = FakeGroq()
    config = _config(fake, token_budget=300)
    cases = [_case("norm-001"), _case("norm-002"), _case("bl-d01")]
    with pytest.raises(TokenBudgetError, match=r"stopped after 1/3 cases: token budget") as caught:
        await run_suite(cases, config)
    assert [r.case_id for r in caught.value.results] == ["norm-001"]
    assert caught.value.planned == 3
    assert config.usage is not None
    first_case_tokens = config.usage.tokens
    assert first_case_tokens == sum(100 + i + 10 for i in range(1, fake.answered + 1))
    assert first_case_tokens <= 300  # the next case (about as costly) would have crossed it


async def test_a_large_budget_does_not_stop_the_run() -> None:
    config = _config(FakeGroq(), token_budget=1_000_000)
    results = await run_suite([_case("norm-001"), _case("norm-002")], config)
    assert len(results) == 2
    assert over_budget(config, results) is None
