"""Runtime configuration, read from the environment (or `.env`). No secrets are hardcoded."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["openai", "google", "anthropic", "mock"]
EmbeddingProvider = Literal["local", "openai", "google"]
DataMode = Literal["fixture", "live"]
MemoryBackendName = Literal["sqlite", "okf"]
TracingBackend = Literal["langfuse", "langsmith", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="forbid",
        frozen=True,
    )

    # LLM
    llm_provider: LLMProvider = "mock"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str | None = None

    # Data sources
    data_mode: DataMode = "fixture"
    max_hops: int = Field(default=4, ge=1, le=16)
    tool_timeout_s: float = Field(default=10.0, gt=0, le=300)

    # Storage
    qdrant_url: str = "http://localhost:6333"
    checkpoint_db: Path = Path("./.data/checkpoints.sqlite")
    memory_backend: MemoryBackendName = "sqlite"
    memory_db: Path = Path("./.data/memory.sqlite")
    okf_dir: Path = Path("./.data/okf")

    # Tracing
    tracing_backend: TracingBackend = "none"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "sentinel"

    # Live connectors
    notion_token: SecretStr | None = None
    jira_base_url: str | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None
    gmail_mcp_command: str | None = None

    # Guardrails & permissions
    allowed_email_domains: Annotated[tuple[str, ...], NoDecode] = ("acme.test",)
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    audit_log: Path = Path("./.data/audit.jsonl")

    @field_validator("allowed_email_domains", mode="before")
    @classmethod
    def _split_domains(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list | tuple):
            domains = tuple(str(d).strip().lower().lstrip("@") for d in value if str(d).strip())
            if not domains:
                raise ValueError("ALLOWED_EMAIL_DOMAINS must list at least one domain")
            return domains
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests should construct `Settings(...)` directly instead."""
    return Settings()
