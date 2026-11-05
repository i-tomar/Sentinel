from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.config import Settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_defaults_need_no_credentials() -> None:
    s = Settings()
    assert s.llm_provider == "mock"
    assert s.data_mode == "fixture"
    assert s.tracing_backend == "none"
    assert s.max_hops == 4
    assert s.allowed_email_domains == ("acme.test",)
    assert s.llm_api_key is None


def test_env_example_loads_and_covers_every_setting(isolated_env: Path) -> None:
    shutil.copy(REPO_ROOT / ".env.example", isolated_env / ".env")
    s = Settings()
    assert s.llm_provider == "mock"

    keys = {
        line.split("=", 1)[0].strip()
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert keys == {name.upper() for name in Settings.model_fields}


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MAX_HOPS", "6")
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", " Acme.test, @partner.test ,")
    s = Settings()
    assert s.llm_provider == "openai"
    assert s.max_hops == 6
    assert s.allowed_email_domains == ("acme.test", "partner.test")


def test_empty_env_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("MAX_HOPS", "")
    s = Settings()
    assert s.llm_model is None
    assert s.max_hops == 4


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LLM_PROVIDER", "cohere"),
        ("DATA_MODE", "prod"),
        ("MAX_HOPS", "0"),
        ("INJECTION_THRESHOLD", "1.5"),
        ("TOOL_TIMEOUT_S", "-1"),
        ("ALLOWED_EMAIL_DOMAINS", " , "),
    ],
)
def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings()


def test_unknown_dotenv_key_is_rejected(isolated_env: Path) -> None:
    (isolated_env / ".env").write_text("LLM_PROVDER=openai\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "env",
    [
        {"TRACING_BACKEND": "langfuse"},
        {"TRACING_BACKEND": "langfuse", "LANGFUSE_PUBLIC_KEY": "pk-lf-x"},
        {"TRACING_BACKEND": "langsmith"},
    ],
)
def test_tracing_backend_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError, match="TRACING_BACKEND"):
        Settings()


def test_langfuse_enabled_through_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACING_BACKEND", "langfuse")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    s = Settings()
    assert s.tracing_backend == "langfuse"
    assert s.langfuse_host == "http://127.0.0.1:3000"


def test_secrets_are_masked_in_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-very-secret")
    s = Settings()
    assert s.llm_api_key is not None
    assert s.llm_api_key.get_secret_value() == "sk-very-secret"
    assert "sk-very-secret" not in repr(s)
    assert "sk-very-secret" not in s.model_dump_json()
