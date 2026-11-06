"""Agent-side long-term memory logic: writes, the unverified-claim guard, refresh calls, prompt."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from sentinel.citations import Citation
from sentinel.graph.nodes.answerer import build_messages
from sentinel.graph.nodes.citation_check import check_answer
from sentinel.graph.nodes.execute_tools import execute_calls
from sentinel.graph.nodes.save_memory import (
    apply_writes,
    fact_from_answer,
    flag_conflicts,
    propose_writes,
)
from sentinel.graph.nodes.tool_router import MEMORY_BUCKET, memory_refresh_calls
from sentinel.graph.state import CitationReport, Evidence
from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.retrieval.models import Source

pytestmark = pytest.mark.unit

T = datetime(2026, 10, 27, tzinfo=UTC)


def _note(key: str, value: str, **kw: Any) -> MemoryItem:
    return MemoryItem.model_validate(
        {"key": key, "value": value, "source": "user", "updated_at": T, **kw}
    )


def _evidence(n: int, key: str, text: str) -> Evidence:
    source, doc_id = key.split(":", 1)
    citation = Citation.model_validate(
        {
            "n": n,
            "source": source,
            "doc_id": doc_id,
            "title": doc_id,
            "url": f"https://x/{doc_id}",
            "snippet": text[:40],
            "updated_at": T,
            "fetched_at": T,
        }
    )
    return Evidence(citation=citation, text=text, score=1.0)


V2 = _evidence(1, "notion:auth-migration-v2", "Owner: Priya Sharma. Cutover date: 2026-11-20.")


# --- facts from answers -------------------------------------------------------------------


def test_verified_fact_from_a_clean_cited_answer() -> None:
    answer = "Priya Sharma owns the plan [1]. Cutover date is 2026-11-20 [1]."
    fact = fact_from_answer(
        "Who owns the Q3 auth migration plan v2?", answer, [V2.citation], CitationReport(), [V2]
    )
    assert fact is not None
    assert fact.key == "owns-q3-auth-migration-plan-v2"
    assert fact.value == "Priya Sharma owns the plan. Cutover date is 2026-11-20."
    assert (fact.source, fact.verified, fact.stale) == ("notion:auth-migration-v2", True, False)
    assert fact.updated_at == T
    assert fact.record_keys == ["notion:auth-migration-v2"]
    assert 0 < fact.confidence <= 1


@pytest.mark.parametrize(
    ("answer", "report"),
    [
        ("I don't have enough evidence to answer this.", CitationReport()),
        ("Priya owns it [1].", CitationReport(uncited_sentences=["It is late."])),
        ("Priya owns it [1].", CitationReport(unverified_memory_claims=["x"])),
    ],
)
def test_no_fact_from_insufficient_or_flagged_answers(answer: str, report: CitationReport) -> None:
    assert fact_from_answer("q", answer, [V2.citation], report, [V2]) is None
    assert fact_from_answer("q", "Priya owns it [1].", [], CitationReport(), [V2]) is None


def test_writes_are_pending_unless_allowed() -> None:
    fact = _note("k", "Priya owns it", source="notion:auth-migration-v2", verified=True)
    stale = _note(
        "old",
        "Vikram owns it",
        source="notion:auth-migration-v1",
        verified=True,
        stale=True,
        stale_reason="superseded",
        superseded_by="notion:auth-migration-v2",
    )
    writes = propose_writes([stale], fact, None)
    assert [(w.op, w.status) for w in writes] == [
        ("mark_stale", "pending_approval"),
        ("put", "pending_approval"),
    ]

    backend = SQLiteMemoryBackend(":memory:")
    backend.put(stale.model_copy(update={"stale": False}))
    pending = apply_writes(writes, backend, "require_approval")
    assert [w.status for w in pending] == ["pending_approval", "pending_approval"]
    assert backend.get("k") is None
    assert backend.get("old").stale is False  # type: ignore[union-attr]

    written = apply_writes(writes, backend, "allow")
    assert [w.status for w in written] == ["written", "written"]
    assert backend.get("k") == fact
    assert backend.get("old").superseded_by == "notion:auth-migration-v2"  # type: ignore[union-attr]

    assert [w.status for w in apply_writes(writes, None, "allow")] == ["pending_approval"] * 2


def test_unchanged_fact_is_skipped() -> None:
    fact = _note("k", "Priya owns it", source="notion:auth-migration-v2", verified=True)
    [write] = propose_writes([], fact, fact)
    assert (write.status, write.reason) == ("skipped", "already remembered")


def test_flag_conflicts_marks_only_conflicting_notes() -> None:
    a, b = _note("a", "x"), _note("b", "y")
    flagged = flag_conflicts([a, b], {"b": ("changed", "jira:PROJ-1")})
    assert [(n.key, n.stale, n.superseded_by) for n in flagged] == [
        ("a", False, None),
        ("b", True, "jira:PROJ-1"),
    ]


# --- unverified memory is never stated as fact --------------------------------------------


def test_sentence_backed_only_by_unverified_memory_is_removed() -> None:
    unverified = ["The auth cutover moved to 2027-03-01"]
    draft = "Priya Sharma is the owner [1]. The auth cutover moved to 2027-03-01 [1]."
    answer, used, report = check_answer(draft, [V2], unverified)
    assert report.unverified_memory_claims == ["The auth cutover moved to 2027-03-01 [1]."]
    assert "2027-03-01" not in answer
    assert "only unverified memory supports" in answer
    assert [c.n for c in used] == [1]
    uncited, _, report2 = check_answer("The auth cutover moved to 2027-03-01.", [V2], unverified)
    assert report2.unverified_memory_claims
    assert "2027-03-01" not in uncited.split("(Removed")[0]


def test_memory_claim_also_supported_by_evidence_is_kept() -> None:
    draft = "Cutover date is 2026-11-20 [1]."
    answer, _, report = check_answer(draft, [V2], ["Cutover date is 2026-11-20"])
    assert report.ok
    assert answer == draft


# --- refresh calls and prompt -------------------------------------------------------------


def test_refresh_calls_only_for_fresh_verified_record_notes() -> None:
    notes = [
        _note("a", "x", source="notion:auth-migration-v2", verified=True),
        _note("b", "x", source="notion:auth-migration-v2", verified=True),  # duplicate record
        _note("c", "x", source="jira:PROJ-142", verified=False),
        _note("d", "x", source="jira:PROJ-147", verified=True, stale=True),
        _note("e", "x", verified=True),  # user-asserted: no record to re-read
        _note("f", "x", source="gmail:thr-0003", verified=True),
    ]
    calls = memory_refresh_calls(notes, hop=1)
    assert [(c.source, c.tool, c.args) for c in calls] == [
        ("notion", "get_page", {"page_id": "auth-migration-v2"}),
        ("gmail", "get_thread", {"thread_id": "thr-0003"}),
    ]
    assert {c.sub_question_id for c in calls} == {MEMORY_BUCKET}


@pytest.mark.anyio
async def test_direct_get_call_is_executed_and_refreshes_known_copy() -> None:
    page = {
        "id": "p1",
        "title": "Plan",
        "url": "u",
        "owner": "a@acme.test",
        "updated_at": T.isoformat(),
        "tags": [],
        "body": "fresh body",
    }

    class Exec:
        async def call(self, source: Source, tool: str, args: dict[str, Any]) -> Any:
            assert (tool, args) == ("get_page", {"page_id": "p1"})
            return page

    [call] = memory_refresh_calls([_note("a", "x", source="notion:p1", verified=True)], hop=1)
    out = await execute_calls([call], Exec(), {}, timeout_s=1, retries=0, backoff_s=0)
    assert out.sq_docs == {MEMORY_BUCKET: ["notion:p1"]}
    assert out.documents["notion:p1"].text == "fresh body"


def test_prompt_labels_notes_and_drops_stale_ones() -> None:
    notes = [
        _note("a", "Priya owns the plan", source="notion:auth-migration-v2", verified=True),
        _note("b", "cutover moved to March"),
        _note("c", "Vikram owns the plan", verified=True, stale=True),
    ]
    _, user = build_messages("Who owns the plan?", [V2], memory_notes=notes)
    body = str(user.content)
    assert '<note verified="true" source="notion:auth-migration-v2"' in body
    assert '<note verified="false" source="user"' in body
    assert "Vikram" not in body
    assert "not evidence; never cite" in body
