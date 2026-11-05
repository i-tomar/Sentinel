from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from sentinel.retrieval.chunking import chunk_document, chunk_id, split_text
from sentinel.retrieval.models import Document

pytestmark = pytest.mark.unit

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def _doc(text: str, doc_id: str = "p1") -> Document:
    return Document(
        source="notion",
        id=doc_id,
        url=f"https://x.test/{doc_id}",
        title="Title",
        text=text,
        updated_at=T0,
        fetched_at=T0,
    )


def test_short_text_is_one_chunk() -> None:
    assert split_text("a\nb\nc", max_chars=100, overlap=10) == ["a\nb\nc"]


def test_chunks_respect_max_chars_and_overlap() -> None:
    lines = [f"line {i:03d} " + "x" * 40 for i in range(60)]
    chunks = split_text("\n".join(lines), max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.splitlines()[-1] == nxt.splitlines()[0]  # one shared line of overlap
    covered = {line for c in chunks for line in c.splitlines()}
    assert covered == set(lines)


def test_overlong_line_and_word_are_split() -> None:
    blob = "Q" * 1000
    chunks = split_text(f"intro words here\n{blob}\noutro", max_chars=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks).count("Q") >= 1000


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_text("x", max_chars=max_chars, overlap=overlap)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    doc = _doc("\n".join(f"para {i} " + "y" * 90 for i in range(30)))
    a, b = chunk_document(doc, 300, 50), chunk_document(doc, 300, 50)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id == chunk_id("notion:p1", 0)
    assert [c.index for c in a] == list(range(len(a)))


def test_chunks_carry_source_metadata() -> None:
    [chunk] = chunk_document(_doc("hello"))
    assert chunk.doc_key == "notion:p1"
    assert chunk.url == "https://x.test/p1"
    assert chunk.updated_at == T0
    assert chunk.doc_hash == _doc("hello").content_hash()


def test_empty_body_still_indexes_title() -> None:
    [chunk] = chunk_document(_doc(""))
    assert chunk.text == "Title"


def test_content_hash_ignores_fetched_at_only() -> None:
    doc = _doc("body")
    later = doc.model_copy(update={"fetched_at": datetime(2027, 1, 1, tzinfo=UTC)})
    edited = doc.model_copy(update={"text": "body v2"})
    assert later.content_hash() == doc.content_hash()
    assert edited.content_hash() != doc.content_hash()
