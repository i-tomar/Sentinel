"""Fixture Notion MCP server (read-only): search_pages, get_page."""

from __future__ import annotations

from collections.abc import Sequence

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from servers.fixtures import failure_modes
from servers.fixtures.common import (
    EMPTY_RESULTS,
    READ,
    Limit,
    Offset,
    load_notion,
    rank,
    results_json,
    snippet,
    to_json,
    tokens,
)
from servers.fixtures.models import NotionPage, PageHit

SOURCE = "notion"


def create_server(pages: Sequence[NotionPage] | None = None) -> MCPServer:
    by_id = {p.id: p for p in (load_notion() if pages is None else pages)}
    server = MCPServer("notion-fixture", instructions="Fake Acme Notion workspace (fixture data).")

    @server.tool(annotations=READ, structured_output=False)
    async def search_pages(query: str, limit: Limit = 10, offset: Offset = 0) -> str:
        """Keyword search over page titles and bodies. Returns JSON {results: [...], total}."""
        if (
            override := await failure_modes.apply(SOURCE, "search_pages", empty=EMPTY_RESULTS)
        ) is not None:
            return override
        ranked = rank(
            by_id.values(),
            query,
            text_of=lambda p: (p.title, p.body),
            recency=lambda p: p.updated_at,
            ident=lambda p: p.id,
        )
        q = tokens(query)
        hits = [
            PageHit(
                id=p.id,
                title=p.title,
                url=p.url,
                owner=p.owner,
                updated_at=p.updated_at,
                snippet=snippet(p.body, q),
            )
            for p in ranked[offset : offset + limit]
        ]
        return results_json(hits, total=len(ranked))

    @server.tool(annotations=READ, structured_output=False)
    async def get_page(page_id: str) -> str:
        """Fetch one page with its full body. Returns the page as JSON."""
        if (override := await failure_modes.apply(SOURCE, "get_page", empty="null")) is not None:
            return override
        page = by_id.get(page_id)
        if page is None:
            raise ToolError(f"page not found: {page_id}")
        return to_json(page)

    return server


if __name__ == "__main__":
    failure_modes.allow_crash()
    failure_modes.exit_if_startup_fails(SOURCE)
    create_server().run(transport="stdio")
