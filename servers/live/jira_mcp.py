"""Live Jira Cloud MCP server (read-only): search_issues, get_issue.

Needs JIRA_BASE_URL (https), JIRA_EMAIL, JIRA_API_TOKEN; optional JIRA_PROJECT_KEYS
(comma-separated) restricts every search to those projects.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from sentinel.live.http import LiveAPIError
from sentinel.live.jira import JiraClient
from servers.fixtures.common import READ, Limit, Offset, to_json


def create_server(client: JiraClient | None = None) -> MCPServer:
    if client is None:
        missing = [
            v for v in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN") if not os.environ.get(v)
        ]
        if missing:
            raise SystemExit(f"missing environment variables: {', '.join(missing)}")
        keys = [k.strip() for k in os.environ.get("JIRA_PROJECT_KEYS", "").split(",") if k.strip()]
        client = JiraClient(
            os.environ["JIRA_BASE_URL"],
            os.environ["JIRA_EMAIL"],
            os.environ["JIRA_API_TOKEN"],
            keys,
        )
    jira = client
    server = MCPServer("jira-live", instructions="Live Jira Cloud (read-only).")

    @server.tool(annotations=READ, structured_output=False)
    async def search_issues(
        query: str = "",
        status: str | None = None,
        assignee: str | None = None,
        label: str | None = None,
        blocked_by: str | None = None,
        limit: Limit = 20,
        offset: Offset = 0,
    ) -> str:
        """Search issues; `blocked_by` takes an issue key. Returns JSON {results: [...], total}."""
        try:
            hits, total = await jira.search_issues(
                query, status, assignee, label, blocked_by, limit, offset
            )
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc
        return to_json({"results": hits, "total": total})

    @server.tool(annotations=READ, structured_output=False)
    async def get_issue(key: str) -> str:
        """Fetch one issue with description, links and comments. Returns JSON."""
        try:
            return to_json(await jira.get_issue(key))
        except LiveAPIError as exc:
            raise ToolError(str(exc)) from exc

    return server


if __name__ == "__main__":
    create_server().run()
