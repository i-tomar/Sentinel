"""Start/connect the MCP servers Sentinel reads from.

Fixture mode launches the bundled fake servers as stdio subprocesses. Live mode (real Notion,
Jira, Gmail) arrives in step (i) and fails loudly until then.
"""

from __future__ import annotations

import os
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
_PASSTHROUGH_PREFIX = "SENTINEL_FIXTURE_"


def fixture_server_params(source: Source) -> StdioServerParameters:
    """stdio launch spec; forwards SENTINEL_FIXTURE_* (failure injection, data dir)."""
    env = {k: v for k, v in os.environ.items() if k.startswith(_PASSTHROUGH_PREFIX)}
    return StdioServerParameters(
        command=sys.executable, args=["-m", FIXTURE_MODULES[source]], env=env or None
    )


@asynccontextmanager
async def open_clients(
    settings: Settings, sources: Sequence[Source]
) -> AsyncIterator[dict[Source, Client]]:
    if settings.data_mode == "live":
        raise NotImplementedError("DATA_MODE=live connectors arrive in step (i); use fixture")
    async with AsyncExitStack() as stack:
        clients: dict[Source, Client] = {}
        for source in sources:
            clients[source] = await stack.enter_async_context(Client(fixture_server_params(source)))
        yield clients
