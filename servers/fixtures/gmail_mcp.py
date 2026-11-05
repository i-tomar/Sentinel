"""Fixture Gmail MCP server: search_threads, get_thread (read), create_draft (write),
send (destructive).

Nothing is ever delivered: drafts and "sent" messages live only in this server process's memory.
Permission checks are the agent's job (permission gate), not the server's.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from servers.fixtures import failure_modes
from servers.fixtures.common import (
    DESTRUCTIVE,
    EMPTY_RESULTS,
    READ,
    WRITE,
    Limit,
    load_gmail,
    rank,
    results_json,
    snippet,
    to_json,
    tokens,
)
from servers.fixtures.generate import NOW
from servers.fixtures.models import Draft, GmailThread, ThreadHit

SOURCE = "gmail"


def _text(thread: GmailThread) -> tuple[str, str]:
    parts = [f"{m.sender} {' '.join(m.to)} {' '.join(m.cc)}\n{m.body}" for m in thread.messages]
    return thread.subject, "\n".join(parts)


def _participants(thread: GmailThread) -> tuple[str, ...]:
    people = {a for m in thread.messages for a in (m.sender, *m.to, *m.cc)}
    return tuple(sorted(people))


def _in_range(thread: GmailThread, after: date | None, before: date | None) -> bool:
    return any(
        (after is None or m.sent_at.date() >= after)
        and (before is None or m.sent_at.date() < before)
        for m in thread.messages
    )


def create_server(threads: Sequence[GmailThread] | None = None) -> MCPServer:
    by_id = {t.id: t for t in (load_gmail() if threads is None else threads)}
    drafts: dict[str, Draft] = {}
    sent: list[str] = []
    server = MCPServer("gmail-fixture", instructions="Fake Acme mailbox (fixture data).")

    @server.tool(annotations=READ, structured_output=False)
    async def search_threads(
        query: str = "",
        after: date | None = None,
        before: date | None = None,
        label: str | None = None,
        limit: Limit = 10,
    ) -> str:
        """Keyword search over subjects, bodies and addresses.

        `after` is inclusive and `before` exclusive (YYYY-MM-DD), matched against any message in
        the thread. Returns JSON {results: [...], total}.
        """
        if (
            override := await failure_modes.apply(SOURCE, "search_threads", empty=EMPTY_RESULTS)
        ) is not None:
            return override
        candidates = [
            t
            for t in by_id.values()
            if _in_range(t, after, before) and (label is None or label.lower() in t.labels)
        ]
        ranked = rank(
            candidates,
            query,
            text_of=_text,
            recency=lambda t: t.messages[-1].sent_at,
            ident=lambda t: t.id,
        )
        q = tokens(query)
        hits = [
            ThreadHit(
                id=t.id,
                url=t.url,
                subject=t.subject,
                participants=_participants(t),
                labels=t.labels,
                message_count=len(t.messages),
                last_message_at=t.messages[-1].sent_at,
                snippet=snippet(t.messages[0].body, q),
            )
            for t in ranked[:limit]
        ]
        return results_json(hits, total=len(ranked))

    @server.tool(annotations=READ, structured_output=False)
    async def get_thread(thread_id: str) -> str:
        """Fetch one thread with every message. Returns the thread as JSON."""
        if (override := await failure_modes.apply(SOURCE, "get_thread", empty="null")) is not None:
            return override
        thread = by_id.get(thread_id)
        if thread is None:
            raise ToolError(f"thread not found: {thread_id}")
        return to_json(thread)

    @server.tool(annotations=WRITE, structured_output=False)
    async def create_draft(
        to: list[str], subject: str, body: str, thread_id: str | None = None
    ) -> str:
        """Create an email draft (not sent). WRITE: requires human approval."""
        if (
            override := await failure_modes.apply(SOURCE, "create_draft", empty="null")
        ) is not None:
            return override
        if not to:
            raise ToolError("at least one recipient is required")
        if thread_id is not None and thread_id not in by_id:
            raise ToolError(f"thread not found: {thread_id}")
        draft = Draft(
            draft_id=f"draft-{len(drafts) + 1:04d}",
            to=tuple(a.strip().lower() for a in to),
            subject=subject,
            body=body,
            thread_id=thread_id,
            created_at=NOW,
        )
        drafts[draft.draft_id] = draft
        return to_json(draft)

    @server.tool(annotations=DESTRUCTIVE, structured_output=False)
    async def send(draft_id: str) -> str:
        """Send an existing draft. DESTRUCTIVE: requires approval plus typed confirmation."""
        if (override := await failure_modes.apply(SOURCE, "send", empty="null")) is not None:
            return override
        draft = drafts.pop(draft_id, None)
        if draft is None:
            raise ToolError(f"draft not found: {draft_id}")
        sent.append(draft_id)
        return to_json(
            {"message_id": f"msg-{len(sent):04d}", "draft_id": draft_id, "to": list(draft.to)}
        )

    return server


if __name__ == "__main__":
    create_server().run()
