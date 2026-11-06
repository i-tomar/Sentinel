"""Live API clients against mock transports (no network, no real accounts)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from sentinel.live.http import LiveAPIError, ReadOnlyGuard, guarded_client
from sentinel.live.notion import API as NOTION_API
from sentinel.live.notion import NOTION_VERSION, NotionClient, blocks_text
from sentinel.retrieval.normalize import notion_document
from tests.live_payloads import (
    PAGE_V1,
    PAGE_V2,
    Recorder,
    notion_blocks,
    notion_page,
)

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

NOW = datetime(2026, 11, 2, tzinfo=UTC)


# --- read-only guard ----------------------------------------------------------------------


async def test_guard_blocks_everything_not_allowlisted() -> None:
    rec = Recorder(lambda r: httpx.Response(200, json={}))
    async with guarded_client((("GET", "https://api.test/read/"),), transport=rec.transport) as c:
        assert (await c.get("https://api.test/read/x")).status_code == 200
        for method, url in (
            ("POST", "https://api.test/read/x"),
            ("PATCH", "https://api.test/read/x"),
            ("DELETE", "https://api.test/read/x"),
            ("GET", "https://api.test/write/x"),
            ("GET", "https://evil.test/read/x"),
        ):
            with pytest.raises(LiveAPIError, match="blocked non-allowlisted"):
                await c.request(method, url)
    assert len(rec.requests) == 1  # nothing blocked ever reached the transport
    assert ReadOnlyGuard(rec.transport, []).permits("GET", "https://x") is False


# --- Notion -------------------------------------------------------------------------------


def _notion_route(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == f"{NOTION_API}/search":
        body = json.loads(request.content)
        if body.get("start_cursor") == "cur-2":
            return httpx.Response(
                200,
                json={
                    "results": [notion_page(PAGE_V1, "Plan (v1)", "2026-07-05T00:00:00.000Z")],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    notion_page(PAGE_V2, "Plan (v2)", "2026-10-27T00:00:00.000Z"),
                    notion_page(
                        "2b3c4d5e-6f70-4a8b-9c0d-1e2f3a4b5c6d",
                        "Old",
                        "2026-01-01T00:00:00.000Z",
                        trashed=True,
                    ),
                ],
                "has_more": True,
                "next_cursor": "cur-2",
            },
        )
    if url == f"{NOTION_API}/pages/{PAGE_V2}":
        return httpx.Response(
            200, json=notion_page(PAGE_V2, "Plan (v2)", "2026-10-27T00:00:00.000Z")
        )
    if url.startswith(f"{NOTION_API}/blocks/{PAGE_V2}/children"):
        return httpx.Response(
            200,
            json=notion_blocks(
                ("heading_1", "Q3 Auth Migration Plan (v2)"),
                ("paragraph", "Owner: Priya Sharma."),
                ("bulleted_list_item", "PROJ-142 billing login"),
            ),
        )
    return httpx.Response(404, json={"object": "error", "code": "object_not_found"})


async def test_notion_search_skips_trash_and_walks_cursors() -> None:
    rec = Recorder(_notion_route)
    client = NotionClient("secret_ntn_token", transport=rec.transport)
    first, total = await client.search_pages("auth migration", limit=1, offset=0)
    assert [h["id"] for h in first] == [PAGE_V2]
    assert total == 2  # one fresh result + has_more
    second, total = await client.search_pages("auth migration", limit=1, offset=1)
    assert [h["title"] for h in second] == ["Plan (v1)"]
    assert total == 2
    req = rec.requests[0]
    assert req.headers["Authorization"] == "Bearer secret_ntn_token"
    assert req.headers["Notion-Version"] == NOTION_VERSION
    assert rec.bodies()[0]["filter"] == {"property": "object", "value": "page"}


async def test_notion_page_maps_to_record_shape() -> None:
    client = NotionClient("t", transport=Recorder(_notion_route).transport)
    raw = await client.get_page(PAGE_V2)
    assert (
        raw["body"]
        == "# Q3 Auth Migration Plan (v2)\nOwner: Priya Sharma.\n- PROJ-142 billing login"
    )
    doc = notion_document(raw, NOW)  # same normalizer as fixture mode
    assert (doc.key, doc.title) == (f"notion:{PAGE_V2}", "Plan (v2)")
    assert doc.updated_at.isoformat().startswith("2026-10-27")


async def test_notion_errors_and_bad_ids() -> None:
    client = NotionClient("secret_ntn_token", transport=Recorder(_notion_route).transport)
    with pytest.raises(LiveAPIError, match="invalid Notion page id"):
        await client.get_page("../users")
    with pytest.raises(LiveAPIError, match="404") as info:
        await client.get_page("ffffffff-ffff-4fff-8fff-ffffffffffff")
    assert "secret_ntn_token" not in str(info.value)


def test_blocks_text_ignores_blocks_without_text() -> None:
    assert blocks_text([{"type": "image", "image": {}}, {"type": "divider"}]) == ""
