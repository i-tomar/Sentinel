"""Dense embedders and the keyword (sparse, BM25-style) encoder.

- `HashEmbedder`: deterministic, offline, dependency-free feature hashing. Used by tests and CI;
  it captures lexical overlap only, so it is a stand-in, not a semantic model.
- `FastEmbedEmbedder`: local ONNX sentence embeddings via fastembed (downloads the model once,
  no API key).
- Keyword vectors use saturated term frequency; Qdrant applies IDF server-side (Modifier.IDF),
  which together gives BM25-like scoring for exact terms such as ticket keys.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from sentinel.config import Settings

_TOKEN = re.compile(r"\w+(?:-\w+)*", re.UNICODE)
BM25_K1 = 1.2


def keyword_tokens(text: str) -> list[str]:
    """Lowercased word tokens; hyphenated terms (`proj-142`) are kept whole and also split."""
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if "-" in match:
            tokens.extend(p for p in match.split("-") if p)
    return tokens


def _bucket(token: str, size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


SPARSE_SPACE = 2**31 - 1  # Qdrant sparse indices are uint32


def sparse_document_vector(text: str) -> tuple[list[int], list[float]]:
    counts = Counter(_bucket(t, SPARSE_SPACE) for t in keyword_tokens(text))
    indices = sorted(counts)
    return indices, [counts[i] * (BM25_K1 + 1) / (counts[i] + BM25_K1) for i in indices]


def sparse_query_vector(text: str) -> tuple[list[int], list[float]]:
    indices = sorted({_bucket(t, SPARSE_SPACE) for t in keyword_tokens(text)})
    return indices, [1.0] * len(indices)


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Short slug; part of the Qdrant collection name so models never mix."""
        ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Signed feature hashing of word unigrams and character trigrams, L2-normalized."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        words = keyword_tokens(text)
        features = words + [f"#{w[i : i + 3]}" for w in words for i in range(max(1, len(w) - 2))]
        for feature in features:
            h = _bucket(feature, 2 * self._dim)
            vec[h // 2] += 1.0 if h % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        from fastembed import TextEmbedding  # heavy import (onnxruntime); only when used

        self._model_name = model
        self._model = TextEmbedding(model_name=model)
        self._dim = int(self._model.embedding_size)

    @property
    def name(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self._model_name.lower()).strip("_")

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder()
    if settings.embedding_provider == "local":
        return FastEmbedEmbedder(settings.embedding_model or DEFAULT_LOCAL_MODEL)
    raise NotImplementedError(
        f"EMBEDDING_PROVIDER={settings.embedding_provider} is not implemented yet; "
        "use 'local' (fastembed, no key) or 'hash' (offline, tests)"
    )
