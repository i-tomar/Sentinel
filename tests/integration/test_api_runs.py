"""Demo FastAPI wrapper: fixture/mock only, observed stages, no live overrides."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from evals.datasets import load_cases
from sentinel.api.app import create_app
from sentinel.api.models import MAX_QUESTION_CHARS
from sentinel.api.rate_limit import RateLimiter
from sentinel.api.runtime import DEMO_DATA_MODE, DEMO_LLM, DemoRuntime
from sentinel.api.store import RunStore
from sentinel.config import get_settings
from servers.fixtures.generate import PII_SEEDS

pytestmark = pytest.mark.integration

OWNER_Q = "Who owns the Q3 auth migration plan v2?"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    # One ingest + fixture MCP stack for the module; lifespan owns startup/shutdown.
    with TestClient(create_app(DemoRuntime())) as test_client:
        yield test_client


def _attack_question() -> str:
    case = next(c for c in load_cases() if c.id == "au-d01")
    return case.question


def test_health_reports_fixture_mock(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["llm"] == DEMO_LLM == "mock"
    assert body["data_mode"] == DEMO_DATA_MODE == "fixture"
    assert "checked_at" in body
    assert "X-Request-Id" in res.headers


def test_normal_question_returns_answer_and_citations(client: TestClient) -> None:
    created = client.post("/api/runs", json={"question": OWNER_Q})
    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["mode"] == "fixture"
    assert payload["status"] in ("completed", "citation_failed", "output_redacted")
    run_id = payload["run_id"]

    got = client.get(f"/api/runs/{run_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["run_id"] == run_id
    assert body["answer"]
    assert body["citations"]
    assert body["data_provenance"] == {
        "mode": "fixture",
        "llm": "mock",
        "is_fixture": True,
    }
    assert "tool_calls" not in body
    assert "args" not in json.dumps(body)

    events = client.get(f"/api/runs/{run_id}/events").json()
    stages = [e["stage"] for e in events["events"]]
    assert "input_guard" in stages
    assert "answer" in stages
    assert "citation_check" in stages
    # Full successful path includes these; never invent if absent.
    assert stages[0] == "input_guard"


def test_input_attack_refused_and_later_stages_absent(client: TestClient) -> None:
    attack = _attack_question()
    created = client.post("/api/runs", json={"question": attack})
    assert created.status_code == 201, created.text
    assert created.json()["status"] == "refused"
    run_id = created.json()["run_id"]

    body = client.get(f"/api/runs/{run_id}").json()
    assert body["status"] == "refused"
    assert body["input_check"] is not None
    assert body["input_check"]["blocked"] is True
    assert body["citations"] == []

    events = client.get(f"/api/runs/{run_id}/events").json()
    stages = [e["stage"] for e in events["events"]]
    assert stages == ["input_guard"]
    assert "blocked" in events["events"][0]["summary"]
    assert "plan" not in stages
    assert "answer" not in stages
    assert "retrieve" not in stages


def test_client_cannot_switch_to_live_provider(client: TestClient) -> None:
    # Extra body fields forbidden.
    res = client.post(
        "/api/runs",
        json={"question": OWNER_Q, "llm_provider": "openai", "data_mode": "live"},
    )
    assert res.status_code == 422
    err = res.json()["error"]
    assert err["code"] == "validation_error"
    assert "request_id" in err
    assert "traceback" not in res.text.lower()
    assert "stack" not in res.text.lower()

    # Override headers forbidden.
    res2 = client.post(
        "/api/runs",
        json={"question": OWNER_Q},
        headers={"X-LLM-Provider": "openai", "X-Data-Mode": "live"},
    )
    assert res2.status_code == 400
    assert res2.json()["error"]["code"] == "override_forbidden"


def test_question_length_capped(client: TestClient) -> None:
    res = client.post("/api/runs", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert res.status_code == 422


def test_no_raw_pii_in_responses(client: TestClient) -> None:
    created = client.post("/api/runs", json={"question": OWNER_Q})
    run_id = created.json()["run_id"]
    blob = client.get(f"/api/runs/{run_id}").text + client.get(f"/api/runs/{run_id}/events").text
    for seed in PII_SEEDS:
        assert seed.value not in blob, seed.value


def test_unknown_run_is_not_found(client: TestClient) -> None:
    res = client.get("/api/runs/does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_openapi_json_disabled(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 404


def test_partial_groq_env_still_runs_mock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    get_settings.cache_clear()
    res = client.post("/api/runs", json={"question": OWNER_Q})
    assert res.status_code == 201, res.text
    assert res.json()["mode"] == "fixture"


def test_rate_limit_returns_429_with_retry_after() -> None:
    limiter = RateLimiter(max_calls=1, window_s=60.0)
    with TestClient(create_app(DemoRuntime(), rate_limiter=limiter)) as c:
        assert c.post("/api/runs", json={"question": OWNER_Q}).status_code == 201
        limited = c.post("/api/runs", json={"question": OWNER_Q})
        assert limited.status_code == 429
        err = limited.json()["error"]
        assert err["code"] == "rate_limited"
        assert "request_id" in err
        assert limited.headers.get("Retry-After")
        assert int(limited.headers["Retry-After"]) >= 1


def test_evicted_run_returns_404_envelope() -> None:
    demo = DemoRuntime()
    demo.store = RunStore(max_runs=1)
    with TestClient(create_app(demo)) as c:
        first = c.post("/api/runs", json={"question": OWNER_Q})
        assert first.status_code == 201, first.text
        old_id = first.json()["run_id"]
        second = c.post("/api/runs", json={"question": OWNER_Q})
        assert second.status_code == 201, second.text
        missing = c.get(f"/api/runs/{old_id}")
        assert missing.status_code == 404
        err = missing.json()["error"]
        assert err["code"] == "not_found"
        assert "request_id" in err
        assert c.get(f"/api/runs/{second.json()['run_id']}").status_code == 200
