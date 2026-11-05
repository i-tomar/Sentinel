"""Opt-in: the same ingestion against a real Qdrant server (e.g. `make up`).

Set SENTINEL_TEST_QDRANT_URL=http://localhost:6333. Uses a throwaway collection that the test
creates and deletes; nothing else in the server is touched.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime

import pytest
from mcp import Client
from qdrant_client import QdrantClient

from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.ingest import ingest
from sentinel.retrieval.models import Source
from sentinel.retrieval.store import ChunkStore
from servers.fixtures import gmail_mcp, jira_mcp, notion_mcp

URL = os.environ.get("SENTINEL_TEST_QDRANT_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.skipif(not URL, reason="set SENTINEL_TEST_QDRANT_URL to test a Qdrant server"),
]


@pytest.fixture
def server_store() -> Iterator[ChunkStore]:
    assert URL
    client = QdrantClient(url=URL)
    store = ChunkStore(client, f"sentinel_test_{uuid.uuid4().hex[:8]}", HashEmbedder())
    try:
        yield store
    finally:
        if client.collection_exists(store.collection):
            client.delete_collection(store.collection)
        client.close()


@pytest.fixture
async def clients() -> AsyncIterator[dict[Source, Client]]:
    async with AsyncExitStack() as stack:
        yield {
            "notion": await stack.enter_async_context(Client(notion_mcp.create_server())),
            "jira": await stack.enter_async_context(Client(jira_mcp.create_server())),
            "gmail": await stack.enter_async_context(Client(gmail_mcp.create_server())),
        }


async def test_ingest_reingest_and_search_on_server(
    clients: dict[Source, Client], server_store: ChunkStore
) -> None:
    t1 = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
    first = await ingest(clients, server_store, fetched_at=t1)
    assert sum(r.added for r in first) == 180
    count = server_store.count()
    second = await ingest(clients, server_store, fetched_at=t1)
    assert sum(r.unchanged for r in second) == 180
    assert server_store.count() == count
    assert server_store.search("PROJ-142", 3)[0].chunk.doc_key == "jira:PROJ-142"
    gmail = server_store.search("SSO login failing field team", 3, sources=["gmail"])
    assert gmail[0].chunk.doc_key == "gmail:thr-0003"
