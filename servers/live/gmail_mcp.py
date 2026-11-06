"""Live Gmail MCP server (read-only): search_threads, get_thread.

Needs GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN for an OAuth client
consented with the gmail.readonly scope only (see `sentinel live gmail-auth`).
"""

from __future__ import annotations

import os
from datetime import date

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from sentinel.live.gmail import GmailAuth, GmailClient
from sentinel.live.http import LiveAPIError
from servers.fixtures.common import READ, Limit, Offset, to_json

REQUIRED = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN")


def create_server(client: GmailClient | None = None) -> MCPServer:
    if client is None:
        missing = [v for v in REQUIRED if not os.environ.get(v)]
        if missing:
            raise SystemExit(f"missing environment variables: {', '.join(missing)}")
        client_id, secret, refresh = (os.environ[v] for v in REQUIRED)
        client = GmailClient(GmailAuth(client_id, secret, refresh))
    gmail = client
    server = MCPServer("gmail-live", instructions="Live Gmail mailbox (read-only).")

    @server.tool(annotations=READ, structured_output=False)
    async def search_threads(
        query: str = "",
        after: date | None = None,
        before: date | None = None,
        label: str | None = None,
        limit: Limit = 10,
        offset: Offset = 0,
    ) -> str:
        """Search threads (Gmail query syntax plus date and label filters). Returns JSON."""
        try:
            hits, total = await gmail.search_threads(query, after, before, label, limit, offset)
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc
        return to_json({"results": hits, "total": total})

    @server.tool(annotations=READ, structured_output=False)
    async def get_thread(thread_id: str) -> str:
        """Fetch one thread with every message's headers and plain text. Returns JSON."""
        try:
            return to_json(await gmail.get_thread(thread_id))
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc

    return server


if __name__ == "__main__":
    create_server().run()
