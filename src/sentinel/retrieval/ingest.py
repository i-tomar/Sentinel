"""Pull records over MCP, normalize them into Documents, and sync them into the chunk store.

Only records returned by the source MCP tools are indexed. Fixture ground truth
(`servers/fixtures/data/manifest.json`) and eval datasets are never read here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from mcp import Client
from pydantic import BaseModel

from sentinel.mcp_clients.tools import NORMALIZERS, SOURCE_TOOLS, ToolCallError, call_json
from sentinel.retrieval.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, chunk_document
from sentinel.retrieval.models import Document, Source
from sentinel.retrieval.store import ChunkStore

PAGE_SIZE = 50


class SourceReport(BaseModel):
    source: Source
    documents: int = 0
    chunks_written: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    truncated: bool = False
    """The listing hit the record cap, so deletions were skipped."""
    error: str | None = None


async def fetch_documents(source: Source, client: Client, fetched_at: datetime) -> list[Document]:
    docs, _ = await fetch_documents_capped(source, client, fetched_at)
    return docs


async def fetch_documents_capped(
    source: Source, client: Client, fetched_at: datetime, max_records: int | None = None
) -> tuple[list[Document], bool]:
    """Return (documents, complete). `complete` is False when `max_records` cut the listing."""
    spec = SOURCE_TOOLS[source]
    ids: list[str] = []
    complete = True
    while True:
        page = await call_json(
            client, spec.search_tool, {"query": "", "limit": PAGE_SIZE, "offset": len(ids)}
        )
        batch = [str(r[spec.id_field]) for r in page["results"]]
        ids.extend(batch)
        total = int(page["total"])
        if max_records is not None and len(ids) >= max_records:
            complete = total <= max_records
            ids = ids[:max_records]
            break
        if not batch or len(ids) >= total:
            break
    docs = []
    for record_id in dict.fromkeys(ids):
        raw = await call_json(client, spec.get_tool, {spec.get_arg: record_id})
        docs.append(NORMALIZERS[source](raw, fetched_at))
    return docs, complete


def sync_documents(
    store: ChunkStore,
    source: Source,
    docs: Sequence[Document],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
    complete: bool = True,
) -> SourceReport:
    """Make the index match `docs` for this source: add, update, skip unchanged, delete gone.
    With `complete=False` (a capped listing) nothing is deleted: absence proves nothing."""
    report = SourceReport(source=source, documents=len(docs), truncated=not complete)
    indexed = store.doc_hashes(source)
    unchanged: list[str] = []
    for doc in docs:
        previous = indexed.get(doc.key)
        if previous == doc.content_hash():
            unchanged.append(doc.key)
            continue
        chunks = chunk_document(doc, max_chars, overlap)
        store.replace_document(chunks)
        report.chunks_written += len(chunks)
        if previous is None:
            report.added += 1
        else:
            report.updated += 1
    if docs:
        store.touch(unchanged, docs[0].fetched_at)
    report.unchanged = len(unchanged)
    gone = set(indexed) - {d.key for d in docs} if complete else set()
    store.delete_documents(gone)
    report.deleted = len(gone)
    return report


async def ingest(
    clients: Mapping[Source, Client],
    store: ChunkStore,
    *,
    fetched_at: datetime | None = None,
    max_records: int | None = None,
) -> list[SourceReport]:
    """Fetch and sync each source. A failing source is reported and its index left untouched."""
    store.ensure_collection()
    now = fetched_at or datetime.now(UTC)
    reports = []
    for source, client in clients.items():
        try:
            docs, complete = await fetch_documents_capped(source, client, now, max_records)
        except (ToolCallError, KeyError, TypeError, ValueError) as exc:
            reports.append(SourceReport(source=source, error=f"{type(exc).__name__}: {exc}"))
            continue
        reports.append(sync_documents(store, source, docs, complete=complete))
    return reports
