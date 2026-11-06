from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.config import Settings
from sentinel.memory.freshness import find_conflicts, support
from sentinel.memory.models import MemoryItem, memory_key
from sentinel.memory.policy import decide, make_memory_backend
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.retrieval.models import Document

pytestmark = pytest.mark.unit

T = datetime(2026, 10, 1, tzinfo=UTC)


def _item(key: str, value: str, **kw: object) -> MemoryItem:
    return MemoryItem.model_validate(
        {"key": key, "value": value, "source": "user", "updated_at": T, **kw}
    )


@pytest.fixture(params=["file", "memory"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> SQLiteMemoryBackend:
    path = tmp_path / "memory.sqlite" if request.param == "file" else ":memory:"
    return SQLiteMemoryBackend(path)


# --- model --------------------------------------------------------------------------------


def test_memory_key_from_question() -> None:
    assert memory_key("Who owns the Q3 auth migration plan?") == "owns-q3-auth-migration-plan"
    assert memory_key("???") == "untitled"


@pytest.mark.parametrize(
    "overrides",
    [
        {"key": "Bad Key!"},
        {"source": "slack:C123"},
        {"confidence": 1.5},
        {"value": ""},
    ],
)
def test_item_validation(overrides: dict[str, object]) -> None:
    base = {"key": "k", "value": "v", "source": "user", "updated_at": T}
    with pytest.raises(ValidationError):
        MemoryItem.model_validate({**base, **overrides})


def test_source_record() -> None:
    assert _item("k", "v", source="jira:PROJ-142").source_record == "jira:PROJ-142"
    assert _item("k", "v").source_record is None


# --- get / put / search / mark_stale ------------------------------------------------------


def test_put_get_round_trip_and_upsert(backend: SQLiteMemoryBackend) -> None:
    item = _item(
        "q3-owner",
        "Priya owns it",
        source="notion:auth-migration-v2",
        verified=True,
        confidence=0.9,
        record_keys=["notion:auth-migration-v2"],
    )
    backend.put(item)
    assert backend.get("q3-owner") == item
    backend.put(item.model_copy(update={"value": "Priya Sharma owns it"}))
    assert backend.get("q3-owner").value == "Priya Sharma owns it"  # type: ignore[union-attr]
    assert len(backend.all()) == 1
    assert backend.get("missing") is None


def test_search_ranks_fresh_and_verified_first(backend: SQLiteMemoryBackend) -> None:
    backend.put(_item("plan-owner-unverified", "auth migration owner is Vikram", confidence=0.9))
    backend.put(
        _item(
            "plan-owner-verified",
            "auth migration owner is Priya",
            verified=True,
            confidence=0.6,
            source="notion:auth-migration-v2",
        )
    )
    backend.put(
        _item(
            "plan-owner-stale",
            "auth migration owner is Rahul",
            verified=True,
            stale=True,
            source="notion:auth-migration-v1",
        )
    )
    backend.put(_item("lunch", "team lunch on Friday"))

    hits = backend.search("who is the auth migration owner")
    assert [h.key for h in hits] == ["plan-owner-verified", "plan-owner-unverified"]
    with_stale = backend.search("who is the auth migration owner", include_stale=True)
    assert [h.key for h in with_stale][-1] == "plan-owner-stale"
    assert backend.search("auth migration owner", limit=1)[0].key == "plan-owner-verified"
    assert backend.search("completely unrelated words") == []


def test_mark_stale(backend: SQLiteMemoryBackend) -> None:
    backend.put(_item("k", "value", verified=True, source="jira:PROJ-142"))
    updated = backend.mark_stale("k", "source changed", superseded_by="jira:PROJ-142")
    assert updated is not None
    assert (updated.stale, updated.stale_reason, updated.superseded_by) == (
        True,
        "source changed",
        "jira:PROJ-142",
    )
    assert backend.get("k") == updated
    assert backend.all(include_stale=False) == []
    assert backend.mark_stale("missing", "x") is None


def test_file_backend_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite"
    SQLiteMemoryBackend(path).put(_item("k", "v"))
    assert SQLiteMemoryBackend(path).get("k") is not None


def test_backend_factory(tmp_path: Path) -> None:
    settings = Settings(memory_db=tmp_path / "x.sqlite")
    assert isinstance(make_memory_backend(settings), SQLiteMemoryBackend)
    with pytest.raises(NotImplementedError, match="okf"):
        make_memory_backend(Settings(memory_backend="okf"))


# --- freshness: memory vs newer source records -------------------------------------------


def _doc(key: str, text: str, updated: datetime) -> Document:
    source, doc_id = key.split(":", 1)
    return Document.model_validate(
        {
            "source": source,
            "id": doc_id,
            "url": "u",
            "title": doc_id,
            "text": text,
            "updated_at": updated,
            "fetched_at": updated,
        }
    )


def test_support() -> None:
    assert support("PROJ-142 is Blocked", "Status: Blocked. PROJ-142 billing") == 1.0
    # the shared ticket id must not count as support for a different claim
    assert support("PROJ-142 is In Progress", "Status: Blocked. PROJ-142") == 0.0


def test_newer_source_record_that_disagrees_wins() -> None:
    item = _item(
        "proj-142-status", "PROJ-142 is in progress", source="jira:PROJ-142", verified=True
    )
    newer = _doc("jira:PROJ-142", "PROJ-142 status: Blocked by PROJ-120", T + timedelta(days=5))
    [conflict] = find_conflicts([item], {newer.key: newer}, [])
    assert conflict.key == "proj-142-status"
    assert conflict.superseded_by == "jira:PROJ-142"
    assert "no longer supports" in conflict.reason


def test_newer_record_that_still_agrees_is_not_a_conflict() -> None:
    item = _item("proj-142-status", "PROJ-142 is blocked", source="jira:PROJ-142", verified=True)
    newer = _doc("jira:PROJ-142", "PROJ-142 status: Blocked", T + timedelta(days=5))
    assert find_conflicts([item], {newer.key: newer}, []) == []


def test_older_record_does_not_override_memory() -> None:
    item = _item("k", "PROJ-142 is in progress", source="jira:PROJ-142", verified=True)
    older = _doc("jira:PROJ-142", "Status: Blocked", T - timedelta(days=5))
    assert find_conflicts([item], {older.key: older}, []) == []


def test_superseded_source_record_makes_memory_stale() -> None:
    item = _item(
        "q3-owner", "Vikram Rao owns the plan", source="notion:auth-migration-v1", verified=True
    )
    [conflict] = find_conflicts(
        [item],
        {},
        ["notion:auth-migration-v1"],
        {"notion:auth-migration-v1": "notion:auth-migration-v2"},
    )
    assert conflict.superseded_by == "notion:auth-migration-v2"


def test_items_without_record_source_or_already_stale_are_skipped() -> None:
    items = [_item("a", "x"), _item("b", "y", source="jira:PROJ-1", stale=True)]
    assert find_conflicts(items, {}, ["jira:PROJ-1"]) == []


# --- write policy ---------------------------------------------------------------------------


def test_write_policy() -> None:
    assert decide("live", False) == "require_approval"
    assert decide("fixture", False) == "require_approval"
    assert decide("fixture", True) == "allow"
    with pytest.raises(ValueError, match="only allowed with DATA_MODE=fixture"):
        decide("live", True)


def test_settings_refuse_auto_approve_on_live_data() -> None:
    assert Settings(memory_auto_approve=True).memory_auto_approve
    with pytest.raises(ValidationError, match="MEMORY_AUTO_APPROVE"):
        Settings(data_mode="live", memory_auto_approve=True)
