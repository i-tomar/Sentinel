from __future__ import annotations

import base64
import json
import re
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sentinel import infra

pytestmark = pytest.mark.unit

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
PK, SK = "pk-lf-test", "sk-lf-test"


def test_generated_env_has_strong_unique_secrets() -> None:
    a, b = infra.generate_infra_env(), infra.generate_infra_env()
    assert re.fullmatch(r"[0-9a-f]{64}", a["ENCRYPTION_KEY"])  # Langfuse requires 256-bit hex
    assert a["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"].startswith("pk-lf-")
    assert a["LANGFUSE_INIT_PROJECT_SECRET_KEY"].startswith("sk-lf-")
    for key in ("POSTGRES_PASSWORD", "SALT", "NEXTAUTH_SECRET", "LANGFUSE_INIT_USER_PASSWORD"):
        assert len(a[key]) >= 24
        assert a[key] != b[key]


def test_generated_env_covers_every_compose_variable() -> None:
    required = set(re.findall(r"\$\{(\w+):\?", COMPOSE.read_text(encoding="utf-8")))
    assert required
    assert required <= set(infra.generate_infra_env())


def test_render_parse_roundtrip() -> None:
    values = infra.generate_infra_env()
    assert infra.parse_env(infra.render_env(values)) == values


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / ".env.infra"
    first = infra.write_infra_env(path)
    with pytest.raises(FileExistsError):
        infra.write_infra_env(path)
    assert infra.read_infra_env(path) == first
    assert infra.write_infra_env(path, force=True) != first


def test_app_env_lines_enable_langfuse() -> None:
    lines = infra.app_env_lines(
        {"LANGFUSE_INIT_PROJECT_PUBLIC_KEY": PK, "LANGFUSE_INIT_PROJECT_SECRET_KEY": SK}
    )
    assert lines == [
        "TRACING_BACKEND=langfuse",
        "LANGFUSE_HOST=http://127.0.0.1:3000",
        f"LANGFUSE_PUBLIC_KEY={PK}",
        f"LANGFUSE_SECRET_KEY={SK}",
    ]


# --- health checks against a real local HTTP server ---------------------------------------


class _FakeServices(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        expected = "Basic " + base64.b64encode(f"{PK}:{SK}".encode()).decode()
        if self.path == "/readyz":
            self._send(200, b"all shards are ready")
        elif self.path == "/api/public/health":
            self._send(200, b'{"status":"OK","version":"4.76.0"}')
        elif self.path == "/api/public/projects":
            if self.headers.get("Authorization") == expected:
                self._send(200, json.dumps({"data": [{"name": "sentinel"}]}).encode())
            else:
                self._send(401, b'{"message":"Invalid credentials"}')
        else:
            self._send(404, b"not found")

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServices)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_checks_pass_against_healthy_services(base_url: str) -> None:
    assert infra.check_qdrant(base_url).ok
    health = infra.check_langfuse_health(base_url + "/")
    assert health.ok
    assert "4.76.0" in health.detail
    auth = infra.check_langfuse_auth(base_url, PK, SK)
    assert auth.ok
    assert auth.detail == "keys valid for project: sentinel"


def test_bad_keys_fail_without_leaking_the_secret(base_url: str) -> None:
    auth = infra.check_langfuse_auth(base_url, PK, "sk-lf-wrong")
    assert not auth.ok
    assert "401" in auth.detail
    assert "sk-lf-wrong" not in auth.detail


def test_unreachable_service_fails() -> None:
    result = infra.check_qdrant("http://127.0.0.1:9", timeout=0.5)
    assert not result.ok
    assert result.detail.startswith("unreachable")


def test_non_http_urls_are_refused() -> None:
    result = infra.check_qdrant("file:///etc/passwd")
    assert not result.ok
    assert "non-HTTP" in result.detail


def test_wait_for_retries_until_healthy() -> None:
    outcomes = iter([False, False, True])
    sleeps: list[float] = []

    def run() -> list[infra.CheckResult]:
        return [infra.CheckResult(name="x", ok=next(outcomes), detail="")]

    results = infra.wait_for(run, timeout_s=60, interval_s=2, sleep=sleeps.append)
    assert results[0].ok
    assert sleeps == [2, 2]


def test_wait_for_gives_up_at_deadline() -> None:
    now = [0.0]

    def run() -> list[infra.CheckResult]:
        return [infra.CheckResult(name="x", ok=False, detail="")]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    results = infra.wait_for(run, timeout_s=10, interval_s=4, sleep=sleep, clock=lambda: now[0])
    assert not results[0].ok
    assert now[0] == 12


def _fetch_ok(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    return 200, b"not json"


def test_auth_check_rejects_non_json_success() -> None:
    assert not infra.check_langfuse_auth("http://x", PK, SK, fetch=_fetch_ok).ok
