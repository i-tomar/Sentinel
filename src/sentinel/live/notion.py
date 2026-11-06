"""Read-only Notion API client mapped to Sentinel's Notion record shape.

Allowed requests: POST /v1/search (Notion's search is a POST but reads only) and GETs on
/v1/pages/ and /v1/blocks/. Page bodies are the plain text of top-level blocks (capped).
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from sentinel.live.http import LiveAPIError, guarded_client, request_json

API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
MAX_SEARCH_PAGES = 10
MAX_BLOCKS = 300
_PAGE_ID = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)
ALLOWED = (("POST", f"{API}/search"), ("GET", f"{API}/pages/"), ("GET", f"{API}/blocks/"))
_HEADING_PREFIX = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}
_LIST_PREFIX = {"bulleted_list_item": "- ", "numbered_list_item": "- ", "to_do": "- "}


def plain_text(rich_text: list[dict[str, Any]] | None) -> str:
    return "".join(str(part.get("plain_text", "")) for part in rich_text or [])


def page_title(page: dict[str, Any]) -> str:
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return plain_text(prop.get("title")) or "Untitled"
    return "Untitled"


def blocks_text(blocks: list[dict[str, Any]]) -> str:
    lines = []
    for block in blocks:
        kind = str(block.get("type", ""))
        payload = block.get(kind) or {}
        if not isinstance(payload, dict) or "rich_text" not in payload:
            continue
        text = plain_text(payload.get("rich_text"))
        if text:
            lines.append(_HEADING_PREFIX.get(kind, "") + _LIST_PREFIX.get(kind, "") + text)
    return "\n".join(lines)


def page_hit(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": page["id"],
        "title": page_title(page),
        "url": page.get("url", ""),
        "owner": f"notion-user:{(page.get('created_by') or {}).get('id', 'unknown')}",
        "updated_at": page["last_edited_time"],
        "snippet": "",
    }


class NotionClient:
    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.http = guarded_client(
            ALLOWED,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def search_pages(
        self, query: str, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (hits[offset:offset+limit], total-or-lower-bound). Notion paginates with
        opaque cursors, so offsets are emulated by walking pages (bounded)."""
        wanted = offset + limit
        found: list[dict[str, Any]] = []
        cursor: str | None = None
        has_more = False
        for _ in range(MAX_SEARCH_PAGES):
            body: dict[str, Any] = {
                "query": query,
                "filter": {"property": "object", "value": "page"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor
            data = await request_json(self.http, "POST", f"{API}/search", json=body)
            found += [p for p in data.get("results", []) if not p.get("in_trash")]
            has_more = bool(data.get("has_more"))
            cursor = data.get("next_cursor")
            if not has_more or not cursor or len(found) >= wanted:
                break
        hits = [page_hit(p) for p in found[offset:wanted]]
        return hits, len(found) + (1 if has_more else 0)

    async def get_page(self, page_id: str) -> dict[str, Any]:
        if not _PAGE_ID.match(page_id):
            raise LiveAPIError(f"invalid Notion page id: {page_id!r}")
        page = await request_json(self.http, "GET", f"{API}/pages/{page_id}")
        blocks: list[dict[str, Any]] = []
        cursor: str | None = None
        while len(blocks) < MAX_BLOCKS:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = await request_json(
                self.http, "GET", f"{API}/blocks/{page_id}/children", params=params
            )
            blocks += data.get("results", [])
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
        hit = page_hit(page)
        return {
            "id": hit["id"],
            "title": hit["title"],
            "url": hit["url"],
            "owner": hit["owner"],
            "created_at": page.get("created_time", page["last_edited_time"]),
            "updated_at": page["last_edited_time"],
            "tags": [],
            "body": blocks_text(blocks[:MAX_BLOCKS]),
        }
