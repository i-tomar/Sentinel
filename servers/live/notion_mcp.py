"""Live Notion MCP server (read-only): search_pages, get_page. Needs NOTION_TOKEN."""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from sentinel.live.http import LiveAPIError
from sentinel.live.notion import NotionClient
from servers.fixtures.common import READ, Limit, Offset, to_json


def create_server(client: NotionClient | None = None) -> MCPServer:
    if client is None:
        token = os.environ.get("NOTION_TOKEN")
        if not token:
            raise SystemExit("NOTION_TOKEN is not set")
        client = NotionClient(token)
    notion = client
    server = MCPServer("notion-live", instructions="Live Notion workspace (read-only).")

    @server.tool(annotations=READ, structured_output=False)
    async def search_pages(query: str, limit: Limit = 10, offset: Offset = 0) -> str:
        """Search pages shared with the integration. Returns JSON {results: [...], total}."""
        try:
            hits, total = await notion.search_pages(query, limit, offset)
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc
        return to_json({"results": hits, "total": total})

    @server.tool(annotations=READ, structured_output=False)
    async def get_page(page_id: str) -> str:
        """Fetch one page with the text of its top-level blocks. Returns JSON."""
        try:
            return to_json(await notion.get_page(page_id))
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc

    return server


if __name__ == "__main__":
    create_server().run()
