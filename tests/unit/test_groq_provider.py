"""LLM_PROVIDER=groq on mocked HTTP: request shape, token usage, 429 retries, and the API key
never leaking. No request leaves the process (httpx.MockTransport)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import anyio
import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr, ValidationError

from evals.run import default_baseline
from sentinel.config import Settings
from sentinel.llm import (
    GROQ_BASE_URL,
    make_chat_model,
    rate_limit_policy,
    rate_limited_clients,
)
from sentinel.llm_ratelimit import RateLimitPolicy, RateLimitWaits

pytestmark = pytest.mark.unit

FAKE_KEY = "gsk_test_" + "k3y" * 8
MODEL = "openai/gpt-oss-120b"
NO_WAIT = RateLimitPolicy(backoff_s=0.0)


def _settings(**changes: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "llm_provider": "groq",
        "llm_model": MODEL,
        "llm_api_key": FAKE_KEY,
        **changes,
    }
    return Settings(**values)


def completion(
    content: str = "PROJ-142 is blocked [1].", prompt: int = 120, out: int = 9
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": out,
            "total_tokens": prompt + out,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


Handler = Callable[[httpx.Request], httpx.Response]


def _model(
    handler: Handler, settings: Settings | None = None, policy: RateLimitPolicy = NO_WAIT
) -> tuple[Any, RateLimitWaits]:
    waits = RateLimitWaits()
    mock = httpx.MockTransport(handler)
    clients = rate_limited_clients(waits, policy, sync_inner=mock, async_inner=mock)
    return make_chat_model(settings or _settings(), clients, waits=waits), waits


def _recording(*responses: httpx.Response) -> tuple[Handler, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0) if queue else httpx.Response(200, json=completion())

    return handler, seen


# --- configuration -------------------------------------------------------------------------


def test_groq_needs_a_model_and_a_key() -> None:
    with pytest.raises(ValidationError, match="LLM_MODEL and LLM_API_KEY"):
        Settings(_env_file=None, llm_provider="groq", llm_model=MODEL)
    with pytest.raises(ValidationError, match="LLM_MODEL and LLM_API_KEY"):
        Settings(_env_file=None, llm_provider="groq", llm_api_key=SecretStr(FAKE_KEY))


def test_the_key_is_a_secret_in_settings() -> None:
    settings = _settings()
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings.model_dump())
    assert FAKE_KEY not in settings.model_dump_json()


def test_the_env_example_lists_the_key_empty() -> None:
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    lines = example.read_text(encoding="utf-8").splitlines()
    key_lines = [line for line in lines if line.startswith("LLM_API_KEY=")]
    assert key_lines == ["LLM_API_KEY="]
    assert any("groq" in line for line in lines if line.startswith("LLM_PROVIDER="))


# --- request shape -------------------------------------------------------------------------


def test_request_goes_to_groqs_openai_compatible_endpoint() -> None:
    handler, seen = _recording()
    llm, _ = _model(handler)
    llm.invoke([SystemMessage("rules"), HumanMessage("Which tickets are blocked?")])

    (request,) = seen
    assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    # gpt-oss: reasoning effort set, reasoning text not requested (it could leak into answers)
    assert body["reasoning_effort"] == "low"
    assert body["include_reasoning"] is False
    # fields Groq rejects with HTTP 400
    for unsupported in ("logprobs", "logit_bias", "top_logprobs"):
        assert unsupported not in body
    assert body.get("n", 1) == 1
    assert all("name" not in m for m in body["messages"])


def test_reasoning_options_only_for_gpt_oss_models() -> None:
    handler, seen = _recording()
    llm, _ = _model(handler, _settings(llm_model="some/other-model"))
    llm.invoke([HumanMessage("hi")])
    body = json.loads(seen[0].content)
    assert "reasoning_effort" not in body
    assert "include_reasoning" not in body


def test_reasoning_effort_is_configurable() -> None:
    handler, seen = _recording()
    llm, _ = _model(handler, _settings(groq_reasoning_effort="medium"))
    llm.invoke([HumanMessage("hi")])
    assert json.loads(seen[0].content)["reasoning_effort"] == "medium"


def test_the_judge_model_override() -> None:
    handler, seen = _recording()
    mock = httpx.MockTransport(handler)
    clients = rate_limited_clients(RateLimitWaits(), NO_WAIT, sync_inner=mock, async_inner=mock)
    judge = make_chat_model(_settings(), clients, model="openai/gpt-oss-20b")
    judge.invoke([HumanMessage("grade")])
    assert json.loads(seen[0].content)["model"] == "openai/gpt-oss-20b"


def test_real_model_baselines_are_per_provider_and_model() -> None:
    path = default_baseline(f"groq:{MODEL}", "all")
    assert path.name == "groq-openai-gpt-oss-120b-all.json"
    assert path.parent.name == "baselines"
    assert not path.exists()  # no real-model baseline is created in this step


# --- token usage ---------------------------------------------------------------------------


def test_token_usage_is_read_from_the_response() -> None:
    handler, _ = _recording(httpx.Response(200, json=completion(prompt=321, out=45)))
    llm, _ = _model(handler)
    reply = llm.invoke([HumanMessage("hi")])
    assert isinstance(reply, AIMessage)
    assert reply.usage_metadata is not None
    assert reply.usage_metadata["input_tokens"] == 321
    assert reply.usage_metadata["output_tokens"] == 45
    assert reply.content == "PROJ-142 is blocked [1]."


def test_token_usage_on_the_async_path() -> None:
    handler, _ = _recording(httpx.Response(200, json=completion(prompt=50, out=5)))
    llm, _ = _model(handler)
    reply = anyio.run(llm.ainvoke, [HumanMessage("hi")])
    assert reply.usage_metadata["input_tokens"] == 50
    assert reply.usage_metadata["output_tokens"] == 5


# --- 429 -----------------------------------------------------------------------------------


def _limited(retry_after: str = "0") -> httpx.Response:
    return httpx.Response(
        429,
        headers={"retry-after": retry_after},
        json={
            "error": {
                "message": "Rate limit reached",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )


def test_a_429_is_retried_by_the_transport_not_the_client() -> None:
    handler, seen = _recording(_limited("0"), _limited("0"))
    llm, waits = _model(handler)
    reply = llm.invoke([HumanMessage("hi")])
    assert reply.content == "PROJ-142 is blocked [1]."
    assert len(seen) == 3
    assert waits.retries == 2
    # the OpenAI client never retried on its own: every attempt carries retry count 0
    assert {r.headers.get("x-stainless-retry-count") for r in seen} == {"0"}


def test_a_429_on_the_async_path() -> None:
    handler, seen = _recording(_limited("0"))
    llm, waits = _model(handler)
    reply = anyio.run(llm.ainvoke, [HumanMessage("hi")])
    assert reply.usage_metadata["input_tokens"] == 120
    assert (len(seen), waits.retries) == (2, 1)


def test_an_exhausted_rate_limit_raises_and_is_marked() -> None:
    handler, seen = _recording(*[_limited("0") for _ in range(10)])
    llm, waits = _model(handler, policy=RateLimitPolicy(retries=2, backoff_s=0.0))
    with pytest.raises(openai.RateLimitError) as caught:
        llm.invoke([HumanMessage("hi")])
    assert len(seen) == 3
    assert waits.exhausted
    assert FAKE_KEY not in str(caught.value)


def test_the_rate_limit_policy_comes_from_settings() -> None:
    policy = rate_limit_policy(_settings(llm_rate_limit_retries=3, llm_rate_limit_max_wait_s=45))
    assert (policy.retries, policy.max_wait_s) == (3, 45.0)
    defaults = rate_limit_policy(_settings())
    assert (defaults.retries, defaults.max_wait_s) == (6, 120.0)


# --- the key never leaks -------------------------------------------------------------------


def test_the_key_is_not_logged_even_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    handler, _ = _recording(_limited("0"))
    with caplog.at_level(logging.DEBUG):
        llm, _ = _model(handler)
        llm.invoke([HumanMessage("hi")])
        anyio.run(llm.ainvoke, [HumanMessage("hi")])
    assert caplog.records  # the client does log at debug level
    assert FAKE_KEY not in caplog.text
    assert all(FAKE_KEY not in str(r.args) for r in caplog.records)


def test_the_key_is_not_in_the_model_repr_or_dump() -> None:
    handler, _ = _recording()
    llm, _ = _model(handler)
    assert FAKE_KEY not in repr(llm)
    assert FAKE_KEY not in str(llm.model_dump())


def test_a_missing_extra_names_the_openai_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from sentinel.llm import MissingProviderError

    real_import = builtins.__import__

    def no_openai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langchain_openai":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    with pytest.raises(MissingProviderError, match="uv sync --extra openai"):
        make_chat_model(_settings())
