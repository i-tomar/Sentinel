"""Runtime configuration, read from the environment (or `.env`). No secrets are hardcoded."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["openai", "google", "anthropic", "mock"]
EmbeddingProvider = Literal["local", "hash", "openai", "google"]
DataMode = Literal["fixture", "live"]
MemoryBackendName = Literal["sqlite", "okf"]
LiveSource = Literal["notion", "jira", "gmail"]
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

    # Storage. Local URLs use 127.0.0.1, not localhost: compose binds IPv4 only, and on Windows
    # localhost tries ::1 first, adding ~2 s to every request.
    qdrant_url: str = "http://127.0.0.1:6333"
    """`http(s)://...` for a server, `:memory:` or a directory path for embedded Qdrant."""
    qdrant_collection: str = Field(default="sentinel_chunks", pattern=r"^[A-Za-z0-9_-]+$")
    checkpoint_db: Path = Path("./.data/checkpoints.sqlite")
    memory_backend: MemoryBackendName = "sqlite"
    memory_db: Path = Path("./.data/memory.sqlite")
    memory_auto_approve: bool = False
    """Let the agent write long-term memory without approval. Fixture/eval data only."""
    fixture_state_dir: Path = Path("./.data/fixture-state")
    """Where fixture servers keep approved writes between commands (generated data untouched).
    Delete the directory to reset the fixture data."""
    okf_dir: Path = Path("./.data/okf")

    # Tracing
    tracing_backend: TracingBackend = "none"
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "sentinel"

    # Live connectors (DATA_MODE=live; read-only)
    live_sources: Annotated[tuple[LiveSource, ...], NoDecode] = ("notion",)
    """Sources that go live; enable them one at a time (notion, then jira, then gmail)."""
    live_max_records: int = Field(default=200, ge=1, le=5000)
    """Per-source cap on records pulled by one live ingest."""
    notion_token: SecretStr | None = None
    jira_base_url: str | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None
    jira_project_keys: Annotated[tuple[str, ...], NoDecode] = ()
    gmail_client_id: str | None = None
    gmail_client_secret: SecretStr | None = None
    gmail_refresh_token: SecretStr | None = None
    gmail_mcp_command: str | None = None
    """Optional: launch this command instead of the built-in read-only Gmail server. It must
    expose the same search_threads / get_thread tools."""

    # Guardrails & permissions
    allowed_email_domains: Annotated[tuple[str, ...], NoDecode] = ("acme.test",)
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieved_guard_mode: Literal["remove", "quote"] = "remove"
    """How flagged spans in retrieved records are neutralized before planning and answering."""
    audit_log: Path = Path("./.data/audit.jsonl")
    approval_db: Path = Path("./.data/approvals.sqlite")
    approval_ttl_s: int = Field(default=900, ge=30, le=86_400)
    """How long a proposed write can wait for approval, and an approval for execution."""

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

    @model_validator(mode="after")
    def _llm_provider_has_credentials(self) -> Settings:
        if self.llm_provider != "mock" and not (self.llm_model and self.llm_api_key):
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider} needs LLM_MODEL and LLM_API_KEY "
                "(or use LLM_PROVIDER=mock)"
            )
        return self

    @field_validator("live_sources", mode="before")
    @classmethod
    def _split_sources(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(s.strip().lower() for s in value.split(",") if s.strip())
        return value

    @field_validator("jira_project_keys", mode="before")
    @classmethod
    def _split_project_keys(cls, value: object) -> object:
        if isinstance(value, str):
            value = tuple(k.strip().upper() for k in value.split(",") if k.strip())
        if isinstance(value, tuple | list):
            bad = [k for k in value if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,9}", str(k))]
            if bad:
                raise ValueError(f"invalid JIRA_PROJECT_KEYS entries: {bad}")
        return value

    def live_missing(self) -> dict[str, list[str]]:
        """Enabled live source -> env vars that are missing or invalid (empty = ready)."""
        need: dict[str, list[str]] = {}
        for source in self.live_sources:
            if source == "notion":
                need[source] = [] if self.notion_token else ["NOTION_TOKEN"]
            elif source == "jira":
                missing = [
                    name
                    for name, value in (
                        ("JIRA_BASE_URL", self.jira_base_url),
                        ("JIRA_EMAIL", self.jira_email),
                        ("JIRA_API_TOKEN", self.jira_api_token),
                    )
                    if not value
                ]
                if self.jira_base_url and not self.jira_base_url.startswith("https://"):
                    missing.append("JIRA_BASE_URL (must start with https://)")
                need[source] = missing
            elif self.gmail_mcp_command:
                need[source] = []
            else:
                need[source] = [
                    name
                    for name, value in (
                        ("GMAIL_CLIENT_ID", self.gmail_client_id),
                        ("GMAIL_CLIENT_SECRET", self.gmail_client_secret),
                        ("GMAIL_REFRESH_TOKEN", self.gmail_refresh_token),
                    )
                    if not value
                ]
        return need

    @model_validator(mode="after")
    def _no_memory_auto_approve_on_live_data(self) -> Settings:
        if self.memory_auto_approve and self.data_mode == "live":
            raise ValueError(
                "MEMORY_AUTO_APPROVE is only allowed with DATA_MODE=fixture; live memory "
                "writes always need approval"
            )
        return self

    @model_validator(mode="after")
    def _tracing_backend_has_credentials(self) -> Settings:
        if self.tracing_backend == "langfuse" and not (
            self.langfuse_public_key and self.langfuse_secret_key
        ):
            raise ValueError(
                "TRACING_BACKEND=langfuse needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
                "(for local Langfuse: `uv run sentinel infra app-env`)"
            )
        if self.tracing_backend == "langsmith" and not self.langsmith_api_key:
            raise ValueError("TRACING_BACKEND=langsmith needs LANGSMITH_API_KEY")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests should construct `Settings(...)` directly instead."""
    return Settings()
