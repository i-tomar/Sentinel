"""Start/connect the MCP servers Sentinel reads from.

Fixture mode launches the bundled fake servers. Live mode launches the bundled read-only
live servers (or GMAIL_MCP_COMMAND for Gmail) for the sources in LIVE_SOURCES only. Each
live subprocess receives only its own source's credentials.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager

from mcp import Client, StdioServerParameters

from sentinel.config import Settings
from sentinel.retrieval.models import Source

FIXTURE_MODULES: dict[Source, str] = {
    "notion": "servers.fixtures.notion_mcp",
    "jira": "servers.fixtures.jira_mcp",
    "gmail": "servers.fixtures.gmail_mcp",
}
LIVE_MODULES: dict[Source, str] = {
    "notion": "servers.live.notion_mcp",
    "jira": "servers.live.jira_mcp",
    "gmail": "servers.live.gmail_mcp",
}
_PASSTHROUGH_PREFIX = "SENTINEL_FIXTURE_"


class LiveConfigError(RuntimeError):
    pass


def fixture_server_params(source: Source) -> StdioServerParameters:
    """stdio launch spec; forwards SENTINEL_FIXTURE_* (failure injection, data dir)."""
    env = {k: v for k, v in os.environ.items() if k.startswith(_PASSTHROUGH_PREFIX)}
    return StdioServerParameters(
        command=sys.executable, args=["-m", FIXTURE_MODULES[source]], env=env or None
    )


def _secret(value: object) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else value)


def live_env(source: Source, settings: Settings) -> dict[str, str]:
    """Only the variables this source's server needs (never other sources' secrets)."""
    values: dict[str, object | None] = {}
    if source == "notion":
        values = {"NOTION_TOKEN": settings.notion_token}
    elif source == "jira":
        values = {
            "JIRA_BASE_URL": settings.jira_base_url,
            "JIRA_EMAIL": settings.jira_email,
            "JIRA_API_TOKEN": settings.jira_api_token,
            "JIRA_PROJECT_KEYS": ",".join(settings.jira_project_keys) or None,
        }
    else:
        values = {
            "GMAIL_CLIENT_ID": settings.gmail_client_id,
            "GMAIL_CLIENT_SECRET": settings.gmail_client_secret,
            "GMAIL_REFRESH_TOKEN": settings.gmail_refresh_token,
        }
    return {k: _secret(v) for k, v in values.items() if v}


def live_server_params(source: Source, settings: Settings) -> StdioServerParameters:
    env = live_env(source, settings)
    if source == "gmail" and settings.gmail_mcp_command:
        parts = shlex.split(settings.gmail_mcp_command, posix=os.name != "nt")
        return StdioServerParameters(command=parts[0], args=parts[1:], env=env or None)
    return StdioServerParameters(
        command=sys.executable, args=["-m", LIVE_MODULES[source]], env=env or None
    )


def active_sources(settings: Settings, requested: Sequence[Source]) -> list[Source]:
    """In live mode only LIVE_SOURCES are queried; fixture mode serves every source."""
    if settings.data_mode != "live":
        return list(requested)
    return [s for s in requested if s in settings.live_sources]


@asynccontextmanager
async def open_clients(
    settings: Settings, sources: Sequence[Source]
) -> AsyncIterator[dict[Source, Client]]:
    chosen = active_sources(settings, sources)
    if settings.data_mode == "live":
        missing = {s: m for s, m in settings.live_missing().items() if m and s in chosen}
        if missing:
            listed = "; ".join(f"{s}: {', '.join(m)}" for s, m in missing.items())
            raise LiveConfigError(f"live sources are not configured ({listed})")
    async with AsyncExitStack() as stack:
        clients: dict[Source, Client] = {}
        for source in chosen:
            params = (
                live_server_params(source, settings)
                if settings.data_mode == "live"
                else fixture_server_params(source)
            )
            clients[source] = await stack.enter_async_context(Client(params))
        yield clients
