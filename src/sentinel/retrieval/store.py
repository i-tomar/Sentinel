"""Qdrant-backed chunk store with hybrid (dense + keyword) search fused by reciprocal rank.

One collection per embedder (`<base>_<embedder.name>`) so vectors from different models never
mix. `QDRANT_URL` may be a server URL, `:memory:`, or a directory for embedded on-disk mode.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models

from sentinel.retrieval.embed import Embedder, sparse_document_vector, sparse_query_vector
from sentinel.retrieval.models import Chunk, SearchHit, SearchMode, Source

DENSE = "dense"
SPARSE = "keyword"
_UPSERT_BATCH = 128
_PREFETCH_FACTOR = 4


_RECORD_ID = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")


def record_id_candidates(query: str) -> list[str]:
    """Hyphenated tokens that may be record IDs (`PROJ-142`, `thr-0003`, `pricing-2026`),
    in original, upper and lower case."""
    found = _RECORD_ID.findall(query)
    return list(dict.fromkeys(v for t in found for v in (t, t.upper(), t.lower())))


def make_qdrant_client(url: str) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url)
    return QdrantClient(path=url)


def _match_any(key: str, values: Iterable[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))


def _match(key: str, value: str | int) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class ChunkStore:
    def __init__(self, client: QdrantClient, base_collection: str, embedder: Embedder) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = f"{base_collection}_{embedder.name}"

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors[DENSE].size if isinstance(vectors, dict) else None
            if size != self.embedder.dim:
                raise RuntimeError(
                    f"collection {self.collection} has dense size {size}, "
                    f"embedder {self.embedder.name} produces {self.embedder.dim}"
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # embedded Qdrant ignores payload indexes, and says so
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        with warnings.catch_warnings():  # embedded Qdrant ignores payload indexes, and says so
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field in ("source", "doc_key", "doc_id"):
                self.client.create_payload_index(
                    self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD
                )

    # --- writes ---------------------------------------------------------------------------

    def replace_document(self, chunks: Sequence[Chunk]) -> None:
        """Atomically-enough swap all chunks of one document (delete old, upsert new)."""
        if not chunks:
            return
        keys = {c.doc_key for c in chunks}
        if len(keys) != 1:
            raise ValueError("replace_document expects chunks of exactly one document")
        self.delete_documents(keys)
        texts = [f"{c.title}\n{c.text}" for c in chunks]
        dense = self.embedder.embed_documents(texts)
        points = []
        for chunk, text, vector in zip(chunks, texts, dense, strict=True):
            indices, values = sparse_document_vector(text)
            payload: dict[str, Any] = chunk.model_dump(mode="json")
            payload["updated_ts"] = chunk.updated_at.timestamp()
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        DENSE: vector,
                        SPARSE: models.SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            )
        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(self.collection, points[start : start + _UPSERT_BATCH], wait=True)

    def delete_documents(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        if keys:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[_match_any("doc_key", keys)])
                ),
                wait=True,
            )

    def touch(self, keys: Iterable[str], fetched_at: datetime) -> None:
        """Record that unchanged documents were re-fetched, without re-embedding them."""
        keys = list(keys)
        if keys:
            self.client.set_payload(
                self.collection,
                payload={"fetched_at": fetched_at.isoformat()},
                points=models.Filter(must=[_match_any("doc_key", keys)]),
                wait=True,
            )

    # --- reads ----------------------------------------------------------------------------

    def doc_hashes(self, source: Source) -> dict[str, str]:
        """doc_key -> content hash for every indexed document of `source`."""
        found: dict[str, str] = {}
        offset: Any = None
        flt = models.Filter(must=[_match("source", source), _match("index", 0)])
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=["doc_key", "doc_hash"],
                with_vectors=False,
            )
            for p in points:
                if p.payload:
                    found[str(p.payload["doc_key"])] = str(p.payload["doc_hash"])
            if offset is None:
                return found

    def count(self, source: Source | None = None) -> int:
        flt = models.Filter(must=[_match("source", source)]) if source else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(_to_chunk(p.payload) for p in points)
            if offset is None:
                return chunks

    def search(
        self,
        query: str,
        k: int = 8,
        *,
        sources: Sequence[Source] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        source_conditions: list[models.Condition] = (
            [_match_any("source", sources)] if sources else []
        )
        flt = models.Filter(must=source_conditions) if source_conditions else None
        indices, values = sparse_query_vector(query)
        sparse = models.SparseVector(indices=indices, values=values)
        if mode == "keyword" and not indices:
            return []
        if mode == "dense":
            result = self.client.query_points(
                self.collection,
                query=self.embedder.embed_query(query),
                using=DENSE,
                query_filter=flt,
                limit=k,
                with_payload=True,
            )
        elif mode == "keyword":
            result = self.client.query_points(
                self.collection,
                query=sparse,
                using=SPARSE,
                query_filter=flt,
                limit=k,
                with_payload=True,
            )
        else:
            prefetch = [
                models.Prefetch(
                    query=self.embedder.embed_query(query),
                    using=DENSE,
                    filter=flt,
                    limit=k * _PREFETCH_FACTOR,
                )
            ]
            if indices:
                prefetch.append(
                    models.Prefetch(
                        query=sparse, using=SPARSE, filter=flt, limit=k * _PREFETCH_FACTOR
                    )
                )
            if ids := record_id_candidates(query):
                # A third ranked list restricted to records the query names exactly, so
                # "PROJ-142" ranks the ticket above emails that merely mention it.
                must: list[models.Condition] = [_match_any("doc_id", ids), *source_conditions]
                prefetch.append(
                    models.Prefetch(
                        query=self.embedder.embed_query(query),
                        using=DENSE,
                        filter=models.Filter(must=must),
                        limit=k,
                    )
                )
            result = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        return [SearchHit(chunk=_to_chunk(p.payload), score=p.score) for p in result.points]


def _to_chunk(payload: dict[str, Any] | None) -> Chunk:
    if payload is None:
        raise ValueError("point has no payload")
    return Chunk.model_validate({k: v for k, v in payload.items() if k != "updated_ts"})
