from __future__ import annotations

import logging
import types
import typing
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from sentinel.citations import Citation
from sentinel.graph.checkpoint import STATE_MODELS, make_serde
from sentinel.graph.state import AgentState, Evidence, SubQuestion, TurnRecord
from sentinel.retrieval.models import Document

pytestmark = pytest.mark.unit

T = datetime(2026, 11, 2, tzinfo=UTC)


def _models_in(annotation: Any, seen: set[type[BaseModel]]) -> None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation not in seen:
            seen.add(annotation)
            for field in annotation.model_fields.values():
                _models_in(field.annotation, seen)
        return
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is not None:
        for arg in typing.get_args(annotation):
            _models_in(arg, seen)


def test_every_model_reachable_from_agent_state_is_allowlisted() -> None:
    """Unlisted types come back from the checkpoint as plain dicts, silently."""
    reachable: set[type[BaseModel]] = set()
    for field in AgentState.model_fields.values():
        _models_in(field.annotation, reachable)
    assert reachable <= set(STATE_MODELS), reachable - set(STATE_MODELS)


def test_models_round_trip_with_their_types(caplog: pytest.LogCaptureFixture) -> None:
    doc = Document(
        source="jira", id="PROJ-142", url="u", title="t", text="x", updated_at=T, fetched_at=T
    )
    citation = Citation(
        n=1,
        source="jira",
        doc_id="PROJ-142",
        title="t",
        url="u",
        snippet="s",
        updated_at=T,
        fetched_at=T,
    )
    sq = SubQuestion(id="sq1", text="q", target_sources=["jira"], query="q")
    turn = TurnRecord(
        question="q",
        answer="a",
        as_of=T,
        plan=[sq],
        unanswered=[],
        sq_docs={"sq1": [doc.key]},
        evidence_keys=[doc.key],
        cited_keys=[doc.key],
        failed_sources=["gmail"],
    )
    serde = make_serde()
    with caplog.at_level(logging.WARNING):
        for value in (
            {doc.key: doc},
            [turn],
            [Evidence(citation=citation, text="x", score=1.0)],
        ):
            restored = serde.loads_typed(serde.dumps_typed(value))
            assert restored == value
    assert not [r for r in caplog.records if "deserializ" in r.getMessage().lower()]


def test_unlisted_types_are_not_revived() -> None:
    class Stranger(BaseModel):
        x: int

    serde = make_serde()
    restored = serde.loads_typed(serde.dumps_typed(Stranger(x=1)))
    assert not isinstance(restored, Stranger)
