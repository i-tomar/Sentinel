"""Fixture Gmail MCP server: search_threads, get_thread (read), create_draft (write),
send (destructive).

Nothing is ever delivered. Drafts and "sent" ids live in this process's memory and, when a
state directory is given (the CLI passes FIXTURE_STATE_DIR), in a local state file.
Permission checks are the agent's job (permission gate), not the server's.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from servers.fixtures import failure_modes
from servers.fixtures.common import (
    DESTRUCTIVE,
    EMPTY_RESULTS,
    READ,
    WRITE,
    Limit,
    Offset,
    load_gmail,
    rank,
    results_json,
    snippet,
    to_json,
    tokens,
)
from servers.fixtures.generate import NOW
from servers.fixtures.models import Draft, GmailThread, ThreadHit
from servers.fixtures.state import FixtureState, state_dir_from_env

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


def create_server(
    threads: Sequence[GmailThread] | None = None, state_dir: Path | None = None
) -> MCPServer:
    by_id = {t.id: t for t in (load_gmail() if threads is None else threads)}
    state = FixtureState(SOURCE, state_dir)
    saved = state.load()
    drafts: dict[str, Draft] = {
        k: Draft.model_validate(v) for k, v in saved.get("drafts", {}).items()
    }
    sent: list[str] = [str(d) for d in saved.get("sent", [])]
    server = MCPServer("gmail-fixture", instructions="Fake Acme mailbox (fixture data).")

    def _save() -> None:
        state.save(
            {"drafts": {k: d.model_dump(mode="json") for k, d in drafts.items()}, "sent": sent}
        )

    @server.tool(annotations=READ, structured_output=False)
    async def search_threads(
        query: str = "",
        after: date | None = None,
        before: date | None = None,
        label: str | None = None,
        limit: Limit = 10,
        offset: Offset = 0,
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
            for t in ranked[offset : offset + limit]
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
            draft_id=f"draft-{len(drafts) + len(sent) + 1:04d}",  # ids are never reused
            to=tuple(a.strip().lower() for a in to),
            subject=subject,
            body=body,
            thread_id=thread_id,
            created_at=NOW,
        )
        drafts[draft.draft_id] = draft
        _save()
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
        _save()
        return to_json(
            {"message_id": f"msg-{len(sent):04d}", "draft_id": draft_id, "to": list(draft.to)}
        )

    return server


if __name__ == "__main__":
    create_server(state_dir=state_dir_from_env()).run(transport="stdio")
