from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.citations import CitationResolver, cited_numbers
from sentinel.retrieval.chunking import chunk_id
from sentinel.retrieval.models import Chunk, Source

pytestmark = pytest.mark.unit

UPDATED = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)
FETCHED = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)


def _chunk(source: Source, doc_id: str, index: int = 0, text: str = "body") -> Chunk:
    key = f"{source}:{doc_id}"
    return Chunk(
        chunk_id=chunk_id(key, index),
        doc_key=key,
        source=source,
        doc_id=doc_id,
        title=f"Title {doc_id}",
        url=f"https://{source}.fixture.acme.test/{doc_id}",
        updated_at=UPDATED,
        fetched_at=FETCHED,
        index=index,
        text=text,
        doc_hash="h",
    )


def test_numbers_are_per_document_and_stable() -> None:
    r = CitationResolver()
    a = r.add(_chunk("jira", "PROJ-142", 0))
    b = r.add(_chunk("notion", "auth-migration-v2"))
    a2 = r.add(_chunk("jira", "PROJ-142", 1, "another chunk"))
    assert (a.n, b.n, a2.n) == (1, 2, 1)
    assert a2.snippet == "body"  # first registered chunk wins
    assert r.resolve(2) == b
    assert r.resolve(3) is None
    assert r.resolve_key("jira:PROJ-142") == a
    assert [c.n for c in r.citations] == [1, 2]


def test_check_flags_fabricated_markers() -> None:
    r = CitationResolver()
    r.add(_chunk("jira", "PROJ-142"))
    r.add(_chunk("gmail", "thr-0003"))
    check = r.check("PROJ-142 is blocked [1]. Escalated by Globex [1][7]. Also [99].")
    assert check.used == [1, 7, 99]
    assert check.unknown == [7, 99]
    assert check.unused == [2]
    assert not check.ok
    assert r.check("Only real evidence [2].").ok


def test_cited_numbers_ignores_non_markers() -> None:
    assert cited_numbers("see [3], [a], [12] and [3] again; list[1:2]") == [3, 12]


def test_render_shows_source_title_id_url_and_times() -> None:
    r = CitationResolver()
    citation = r.add(_chunk("notion", "auth-migration-v2"))
    line = citation.render()
    assert line.startswith("[1] Notion: Title auth-migration-v2 (auth-migration-v2) ")
    assert "https://notion.fixture.acme.test/auth-migration-v2" in line
    assert "updated 2026-10-27" in line
    assert "fetched 2026-11-02T10:00:00+00:00" in line
    assert r.render() == line
    assert r.render([5]) == ""


def test_snippet_is_flattened_and_truncated() -> None:
    long_text = "word\n" * 200
    citation = CitationResolver().add(_chunk("gmail", "thr-0001", text=long_text))
    assert len(citation.snippet) <= 280
    assert "\n" not in citation.snippet
    assert citation.snippet.endswith("...")
