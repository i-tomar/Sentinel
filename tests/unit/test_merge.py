"""Merging partial eval reports: coverage checks and metric recomputation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.merge import MergeError, merge_reports, pool_metrics, validate_parts
from evals.metrics import compute_metrics
from evals.resultsio import write_results
from evals.schema import CaseResult

pytestmark = pytest.mark.unit

DATASET_FILES: dict[str, str] = {
    "adversarial_retrieved.jsonl": "a" * 64,
    "adversarial_user.jsonl": "b" * 64,
    "benign_lookalikes.jsonl": "c" * 64,
    "edge.jsonl": "d" * 64,
    "memory_freshness.jsonl": "e" * 64,
    "normal.jsonl": "f" * 64,
    "permissions.jsonl": "g" * 64,
    "pii.jsonl": "h" * 64,
    "tool_failure.jsonl": "i" * 64,
}


def _case(
    case_id: str,
    *,
    split: str = "dev",
    category: str = "normal",
    answer: str = "PROJ-142",
    expected_facts: list[str] | None = None,
    expected_records: list[str] | None = None,
    retrieved: list[str] | None = None,
    latency_ms: int = 100,
    tool_call_count: int = 1,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> CaseResult:
    facts = expected_facts if expected_facts is not None else ["PROJ-142"]
    records = expected_records if expected_records is not None else ["jira:PROJ-142"]
    return CaseResult(
        case_id=case_id,
        category=category,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        answer=answer,
        expected_facts=facts,
        expected_records=records,
        retrieved=retrieved if retrieved is not None else records,
        returned_records={records[0]: "PROJ-142 is blocked."} if records else {},
        latency_ms=latency_ms,
        tool_call_count=tool_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _metrics_blob(
    results: list[CaseResult],
    *,
    run_id: str,
    commit: str = "7803d20",
    tokens: int = 100,
) -> dict[str, Any]:
    keys = {"jira:PROJ-142", "jira:PROJ-147", "notion:auth-migration-v2"}
    by_split: dict[str, list[CaseResult]] = {
        "all": results,
        "dev": [r for r in results if r.split == "dev"],
        "heldout": [r for r in results if r.split == "heldout"],
    }
    metrics: dict[str, Any] = {
        name: compute_metrics(rs, corpus_keys=keys, k=5).model_dump(mode="json")
        for name, rs in by_split.items()
    }
    return {
        "meta": {
            "model": "groq:openai/gpt-oss-120b",
            "subset": "filtered",
            "commit": commit,
            "dataset_version": "1.0.0",
            "dataset_files": DATASET_FILES,
            "cases": len(results),
            "started_at": "2026-11-16T20:00:00+00:00",
            "run_id": run_id,
            "k": 5,
            "judge": False,
            "provider": {
                "rate_limit_retries": 1,
                "server_retries": 0,
                "waited_s": 1.0,
                "tokens": tokens,
            },
        },
        "metrics": metrics,
        "cases": [
            {
                "id": r.case_id,
                "category": r.category,
                "split": r.split,
                "surface": r.surface,
                "failures": [],
            }
            for r in results
        ],
    }


def test_validate_parts_rejects_wrong_commit_or_duplicate_or_gap() -> None:
    planned = ["a-1", "a-2"]
    ok_meta: dict[str, Any] = {
        "commit": "7803d20",
        "dataset_files": DATASET_FILES,
        "dataset_version": "1.0.0",
        "model": "m",
        "k": 5,
    }
    ok: dict[str, Any] = {
        "meta": ok_meta,
        "cases": [{"id": "a-1", "category": "normal", "split": "dev", "failures": []}],
    }
    other: dict[str, Any] = {
        "meta": dict(ok_meta),
        "cases": [{"id": "a-2", "category": "normal", "split": "dev", "failures": []}],
    }
    validate_parts([ok, other], require_commit="7803d20", planned_ids=planned)

    bad_meta: dict[str, Any] = {**dict(other["meta"]), "commit": "deadbee"}
    bad_commit: dict[str, Any] = {**other, "meta": bad_meta}
    with pytest.raises(MergeError, match="commit"):
        validate_parts([ok, bad_commit], require_commit="7803d20", planned_ids=planned)

    dup: dict[str, Any] = {
        "meta": dict(ok_meta),
        "cases": [{"id": "a-1", "category": "normal", "split": "dev", "failures": []}],
    }
    with pytest.raises(MergeError, match="appears"):
        validate_parts([ok, dup], require_commit="7803d20", planned_ids=planned)

    with pytest.raises(MergeError, match="missing"):
        validate_parts([ok, other], require_commit="7803d20", planned_ids=["a-1", "a-2", "a-3"])


def test_pool_metrics_matches_concatenated_case_results() -> None:
    left = [
        _case("norm-001", answer="PROJ-142", latency_ms=100),
        _case("norm-002", answer="missing", expected_facts=["PROJ-147"], latency_ms=200),
    ]
    right = [_case("norm-003", answer="PROJ-142 PROJ-147", expected_facts=["PROJ-142", "PROJ-147"])]
    keys = {"jira:PROJ-142", "jira:PROJ-147"}
    m_left = compute_metrics(left, corpus_keys=keys, k=5)
    m_right = compute_metrics(right, corpus_keys=keys, k=5)
    pooled = pool_metrics([m_left, m_right])
    direct = compute_metrics([*left, *right], corpus_keys=keys, k=5)
    assert pooled.answer_correct == direct.answer_correct
    assert pooled.cases == direct.cases
    assert pooled.recall_at_k.value == pytest.approx(direct.recall_at_k.value)
    assert pooled.latency.mean_tool_calls == pytest.approx(direct.latency.mean_tool_calls)
    # Percentiles need the raw latencies; pooling leaves them unset.
    assert pooled.latency.p50_ms is None
    assert pooled.latency.p95_ms is None
    assert direct.latency.p50_ms is not None


def test_merge_from_results_jsonl_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import merge as merge_mod

    left_cases = [_case("norm-001"), _case("norm-002", answer="x", expected_facts=["NOPE"])]
    right_cases = [_case("edge-001", category="edge", expected_facts=["PROJ-142"])]
    # Patch load_cases to the tiny planned suite so coverage checks pass.
    planned = [
        type("E", (), {"id": "norm-001", "extra_records": []})(),
        type("E", (), {"id": "norm-002", "extra_records": []})(),
        type("E", (), {"id": "edge-001", "extra_records": []})(),
    ]
    monkeypatch.setattr(merge_mod, "load_cases", lambda _datasets=None: planned)
    monkeypatch.setattr(
        merge_mod,
        "corpus_keys",
        lambda _cases: {"jira:PROJ-142", "jira:PROJ-147"},
    )

    part_a, part_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    for part, results, rid, tokens in (
        (part_a, left_cases, "run-a", 40),
        (part_b, right_cases, "run-b", 60),
    ):
        part.mkdir()
        (part / "metrics.json").write_text(
            json.dumps(_metrics_blob(results, run_id=rid, tokens=tokens)), encoding="utf-8"
        )
        write_results(part, results)

    report = merge_reports(
        [part_a, part_b],
        out=out,
        require_commit="7803d20",
        subset="all",
        baseline_advisory=True,
    )
    assert report["meta"]["subset"] == "all"
    assert report["meta"]["cases"] == 3
    assert report["meta"]["merged"]["note"] == "run in 2 parts on different days"
    assert report["meta"]["merged"]["source"] == "results.jsonl"
    assert {p["run_id"] for p in report["meta"]["merged"]["parts"]} == {"run-a", "run-b"}
    assert report["meta"]["provider"]["tokens"] == 100
    assert [c["id"] for c in report["cases"]] == ["norm-001", "norm-002", "edge-001"]
    # Recomputed from CaseResults, not an average of the two part rates.
    assert report["flat"]["all"]["answer_correct"] == pytest.approx(2 / 3)
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Sentinel eval report")
    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert "MERGED RUN" in report_md
    assert "Owner may approve as a baseline after review." in report_md
    assert (out / "results.jsonl").is_file()


def test_merge_from_aggregates_when_results_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evals import merge as merge_mod

    left_cases = [_case("norm-001"), _case("norm-002", answer="x", expected_facts=["NOPE"])]
    right_cases = [_case("edge-001", category="edge")]
    planned = [
        type("E", (), {"id": "norm-001", "extra_records": []})(),
        type("E", (), {"id": "norm-002", "extra_records": []})(),
        type("E", (), {"id": "edge-001", "extra_records": []})(),
    ]
    monkeypatch.setattr(merge_mod, "load_cases", lambda _datasets=None: planned)

    part_a, part_b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    for part, results, rid, tokens in (
        (part_a, left_cases, "run-a", 40),
        (part_b, right_cases, "run-b", 60),
    ):
        part.mkdir()
        (part / "metrics.json").write_text(
            json.dumps(_metrics_blob(results, run_id=rid, tokens=tokens)), encoding="utf-8"
        )

    report = merge_reports(
        [part_a, part_b],
        out=out,
        require_commit="7803d20",
        subset="all",
        baseline_advisory=True,
    )
    assert report["meta"]["merged"]["source"] == "pooled_sufficient_statistics"
    assert report["flat"]["all"]["answer_correct"] == pytest.approx(2 / 3)
    assert report["meta"]["provider"]["tokens"] == 100
    assert report["metrics"]["all"]["latency"]["p50_ms"] is None
