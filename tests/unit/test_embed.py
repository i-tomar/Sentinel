from __future__ import annotations

import math
import os

import pytest

from sentinel.config import Settings
from sentinel.retrieval.embed import (
    FastEmbedEmbedder,
    HashEmbedder,
    keyword_tokens,
    make_embedder,
    sparse_document_vector,
    sparse_query_vector,
)

pytestmark = pytest.mark.unit


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_keyword_tokens_keep_ticket_keys_whole() -> None:
    assert keyword_tokens("Blocked by PROJ-120, see auth-migration-v2.") == [
        "blocked",
        "by",
        "proj-120",
        "proj",
        "120",
        "see",
        "auth-migration-v2",
        "auth",
        "migration",
        "v2",
    ]


def test_keyword_tokens_handle_devanagari() -> None:
    assert keyword_tokens("चालान INV-IN-2210")[-1] == "2210"


def test_sparse_vectors_are_deterministic_and_saturating() -> None:
    idx1, val1 = sparse_document_vector("proj-142 proj-142 proj-142 login")
    idx2, val2 = sparse_document_vector("login proj-142 proj-142 proj-142")
    assert (idx1, val1) == (idx2, val2)
    assert idx1 == sorted(idx1)
    assert max(val1) < 2.2  # tf saturates toward k1 + 1
    q_idx, q_val = sparse_query_vector("PROJ-142 PROJ-142")
    assert set(q_idx) <= set(idx1)
    assert q_val == [1.0] * len(q_idx)


def test_hash_embedder_is_normalized_and_lexical() -> None:
    emb = HashEmbedder(dim=128)
    [a, b, c] = emb.embed_documents(
        ["auth migration owner", "owner of the auth migration", "team lunch on friday"]
    )
    assert math.isclose(_cos(a, a), 1.0, rel_tol=1e-9)
    assert _cos(a, b) > _cos(a, c)
    assert emb.embed_query("auth migration owner") == a
    assert emb.name == "hash128"
    assert emb.dim == 128


def test_make_embedder() -> None:
    assert isinstance(make_embedder(Settings(embedding_provider="hash")), HashEmbedder)
    with pytest.raises(NotImplementedError, match="openai"):
        make_embedder(Settings(embedding_provider="openai"))


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("SENTINEL_TEST_FASTEMBED") != "1",
    reason="downloads a ~70 MB model; set SENTINEL_TEST_FASTEMBED=1 to run",
)
def test_fastembed_local_model() -> None:
    emb = FastEmbedEmbedder()
    [a, b, c] = emb.embed_documents(
        ["Who owns the auth migration?", "Priya owns the OIDC login migration plan.", "Lunch menu"]
    )
    assert emb.dim == len(a) == 384
    assert emb.name == "baai_bge_small_en_v1_5"
    assert _cos(a, b) > _cos(a, c)
    assert len(emb.embed_query("auth")) == 384
