"""Fixture Jira MCP server: search_issues, get_issue (read), update_issue (write).

Writes change this process's copy and, when a state directory is given (the CLI passes
FIXTURE_STATE_DIR), are saved there for later processes. The generated JSON files are never
modified.
Permission checks are the agent's job (permission gate), not the server's.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from servers.fixtures import failure_modes
from servers.fixtures.common import (
    EMPTY_RESULTS,
    READ,
    WRITE,
    Limit,
    Offset,
    load_jira,
    rank,
    results_json,
    to_json,
)
from servers.fixtures.generate import NOW
from servers.fixtures.models import IssueHit, IssueStatus, JiraComment, JiraIssue
from servers.fixtures.state import FixtureState, state_dir_from_env

SOURCE = "jira"


def _text(issue: JiraIssue) -> tuple[str, str]:
    body = "\n".join([issue.key, issue.description, *(c.body for c in issue.comments)])
    return issue.summary, body


def create_server(
    issues: Sequence[JiraIssue] | None = None, state_dir: Path | None = None
) -> MCPServer:
    by_key = {i.key: i for i in (load_jira() if issues is None else issues)}
    state = FixtureState(SOURCE, state_dir)
    saved: dict[str, object] = state.load().get("issues", {})
    for key, raw in saved.items():
        by_key[key] = JiraIssue.model_validate(raw)
    changed: set[str] = set(saved)
    server = MCPServer("jira-fixture", instructions="Fake Acme Jira project PROJ (fixture data).")

    def _get(key: str) -> JiraIssue:
        issue = by_key.get(key.strip().upper())
        if issue is None:
            raise ToolError(f"issue not found: {key}")
        return issue

    @server.tool(annotations=READ, structured_output=False)
    async def search_issues(
        query: str = "",
        status: IssueStatus | None = None,
        assignee: str | None = None,
        label: str | None = None,
        blocked_by: str | None = None,
        limit: Limit = 20,
        offset: Offset = 0,
    ) -> str:
        """Search issues by keywords plus optional exact filters.

        `blocked_by` takes an issue key and returns issues blocked by it. Returns JSON
        {results: [...], total}.
        """
        if (
            override := await failure_modes.apply(SOURCE, "search_issues", empty=EMPTY_RESULTS)
        ) is not None:
            return override
        candidates = [
            i
            for i in by_key.values()
            if (status is None or i.status == status)
            and (assignee is None or i.assignee == assignee.lower())
            and (label is None or label.lower() in i.labels)
            and (blocked_by is None or blocked_by.upper() in i.blocked_by)
        ]
        ranked = rank(
            candidates, query, text_of=_text, recency=lambda i: i.updated_at, ident=lambda i: i.key
        )
        hits = [
            IssueHit(
                key=i.key,
                url=i.url,
                summary=i.summary,
                status=i.status,
                priority=i.priority,
                assignee=i.assignee,
                blocked_by=i.blocked_by,
                updated_at=i.updated_at,
            )
            for i in ranked[offset : offset + limit]
        ]
        return results_json(hits, total=len(ranked))

    @server.tool(annotations=READ, structured_output=False)
    async def get_issue(key: str) -> str:
        """Fetch one issue with description, links and comments. Returns the issue as JSON."""
        if (override := await failure_modes.apply(SOURCE, "get_issue", empty="null")) is not None:
            return override
        return to_json(_get(key))

    @server.tool(annotations=WRITE, structured_output=False)
    async def update_issue(
        key: str,
        status: IssueStatus | None = None,
        assignee: str | None = None,
        comment: str | None = None,
    ) -> str:
        """Change status/assignee and/or add a comment. WRITE: requires human approval."""
        if (
            override := await failure_modes.apply(SOURCE, "update_issue", empty="null")
        ) is not None:
            return override
        issue = _get(key)
        if status is None and assignee is None and comment is None:
            raise ToolError("nothing to update: pass status, assignee and/or comment")
        comments = issue.comments
        if comment:
            comments = (
                *comments,
                JiraComment(
                    id=f"{issue.key}-c{len(comments) + 1}",
                    author="sentinel-agent@acme.test",
                    created_at=NOW,
                    body=comment,
                ),
            )
        updated = issue.model_copy(
            update={
                "status": status or issue.status,
                "assignee": assignee.lower() if assignee else issue.assignee,
                "comments": comments,
                "updated_at": NOW,
            }
        )
        by_key[issue.key] = updated
        changed.add(issue.key)
        state.save({"issues": {k: by_key[k].model_dump(mode="json") for k in sorted(changed)}})
        return to_json(updated)

    return server


if __name__ == "__main__":
    create_server(state_dir=state_dir_from_env()).run(transport="stdio")
