from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import QdrantClient

from sentinel.retrieval.chunking import chunk_document
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.models import Document, Source
from sentinel.retrieval.store import ChunkStore, make_qdrant_client, record_id_candidates

pytestmark = pytest.mark.unit

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _doc(source: Source, doc_id: str, text: str, title: str = "") -> Document:
    return Document(
        source=source,
        id=doc_id,
        url=f"https://{source}.test/{doc_id}",
        title=title or doc_id,
        text=text,
        updated_at=T0,
        fetched_at=T0,
    )


@pytest.fixture
def store() -> ChunkStore:
    s = ChunkStore(QdrantClient(location=":memory:"), "t", HashEmbedder(dim=64))
    s.ensure_collection()
    return s


def test_collection_name_includes_embedder(store: ChunkStore) -> None:
    assert store.collection == "t_hash64"
    store.ensure_collection()  # idempotent


def test_dimension_mismatch_is_refused(store: ChunkStore) -> None:
    other = ChunkStore(store.client, "t", HashEmbedder(dim=64))
    other.embedder = HashEmbedder(dim=32)
    with pytest.raises(RuntimeError, match="dense size 64"):
        other.ensure_collection()


def test_replace_document_removes_stale_chunks(store: ChunkStore) -> None:
    long = _doc("notion", "p", "\n".join(f"line {i} " + "z" * 60 for i in range(40)))
    store.replace_document(chunk_document(long, 300, 30))
    assert store.count() > 3
    short = long.model_copy(update={"text": "now short"})
    store.replace_document(chunk_document(short, 300, 30))
    assert store.count() == 1
    assert store.all_chunks()[0].text == "now short"


def test_replace_requires_single_document(store: ChunkStore) -> None:
    chunks = chunk_document(_doc("jira", "A", "a")) + chunk_document(_doc("jira", "B", "b"))
    with pytest.raises(ValueError, match="exactly one document"):
        store.replace_document(chunks)


def test_hashes_delete_and_touch(store: ChunkStore) -> None:
    a, b = _doc("jira", "PROJ-1", "alpha"), _doc("gmail", "thr-1", "beta")
    for d in (a, b):
        store.replace_document(chunk_document(d))
    assert store.doc_hashes("jira") == {"jira:PROJ-1": a.content_hash()}
    later = T0 + timedelta(days=1)
    store.touch(["jira:PROJ-1"], later)
    touched = next(c for c in store.all_chunks() if c.doc_key == "jira:PROJ-1")
    assert touched.fetched_at == later
    store.delete_documents(["jira:PROJ-1"])
    assert store.count("jira") == 0
    assert store.count("gmail") == 1


def test_search_modes_and_source_filter(store: ChunkStore) -> None:
    docs = [
        _doc("jira", "PROJ-142", "Migrate billing portal login to OIDC", "PROJ-142: billing"),
        _doc("jira", "PROJ-147", "Mobile app SSO login via OIDC", "PROJ-147: mobile"),
        _doc("notion", "lunch", "Team lunch on Friday at noon", "Lunch"),
        _doc("gmail", "thr-1", "Customer says PROJ-142 blocks invoice downloads", "Locked out"),
    ]
    for d in docs:
        store.replace_document(chunk_document(d))
    for mode in ("hybrid", "keyword", "dense"):
        hits = store.search("PROJ-142", 3, mode=mode)
        assert hits[0].chunk.doc_key in {"jira:PROJ-142", "gmail:thr-1"}, mode
        assert "notion:lunch" not in {h.chunk.doc_key for h in hits[:2]}
    gmail_only = store.search("PROJ-142 login", 5, sources=["gmail"])
    assert {h.chunk.source for h in gmail_only} == {"gmail"}
    assert store.search("!!!", 3, mode="keyword") == []
    with pytest.raises(ValueError, match="k must be"):
        store.search("x", 0)


def test_record_id_candidates() -> None:
    assert record_id_candidates("who owns proj-142 and thr-0003?") == [
        "proj-142",
        "PROJ-142",
        "thr-0003",
        "THR-0003",
    ]
    assert record_id_candidates("auth migration plan") == []


def test_named_record_outranks_records_that_mention_it(store: ChunkStore) -> None:
    ticket = _doc("jira", "PROJ-142", "PROJ-142: Billing login\nStatus: Blocked.", "PROJ-142: Bill")
    store.replace_document(chunk_document(ticket))
    store.replace_document(
        chunk_document(
            _doc("gmail", "thr-1", "PROJ-142 PROJ-142 PROJ-142 status update", "PROJ-142 status")
        )
    )
    assert store.search("PROJ-142", 2)[0].chunk.doc_key == "jira:PROJ-142"
    assert store.search("proj-142", 2, sources=["gmail"])[0].chunk.doc_key == "gmail:thr-1"


def test_make_qdrant_client_modes(tmp_path: object) -> None:
    assert isinstance(make_qdrant_client(":memory:"), QdrantClient)
    embedded = make_qdrant_client(str(tmp_path))
    s = ChunkStore(embedded, "disk", HashEmbedder(dim=16))
    s.ensure_collection()
    s.replace_document(chunk_document(_doc("notion", "p", "persisted")))
    assert s.count() == 1
    embedded.close()
