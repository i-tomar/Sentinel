"""The harness runs dataset cases through the real graph on fixture servers (mock LLM) and
fills results the metrics, report and gate can use. No network, no credentials, no baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evals import run as eval_run
from evals.datasets import load_cases
from evals.gate import evaluate_gate
from evals.harness import corpus_keys, run_case, run_suite
from evals.metrics import compute_metrics
from evals.report import build_report, render_markdown
from evals.schema import EvalCase
from servers.fixtures import failure_modes

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

CASES = {c.id: c for c in load_cases()}


def _case(case_id: str) -> EvalCase:
    return CASES[case_id]


async def test_multi_hop_case_fills_retrieval_and_citations() -> None:
    result = await run_case(_case("norm-003"))
    assert result.harness_error is None
    assert {"jira:PROJ-142", "jira:PROJ-147", "jira:PROJ-151"} <= set(result.retrieved)
    assert result.citations
    assert all(c.key in result.returned_records for c in result.citations)
    assert result.tool_call_count > 0
    assert result.input_tokens is None  # the mock reports no real tokens


async def test_attack_permission_and_failure_cases() -> None:
    attack = await run_case(_case("au-d01"))
    assert attack.refused
    poisoned = await run_case(_case("ar-d01"))
    assert "gmail:thr-0012" in poisoned.guard_flagged
    write = await run_case(_case("perm-001"))
    assert write.approval_requested
    assert all(w.approved for w in write.executed_writes)  # only flag-approved memory writes
    outside = await run_case(_case("perm-006"))
    assert outside.policy_denied
    assert not outside.approval_requested
    failure = await run_case(_case("tf-001"))
    assert "I could not read gmail" in failure.answer
    assert os.environ.get(failure_modes.FAILURES_ENV) in (None, "")  # restored afterwards


async def test_pii_case_scans_every_output_path() -> None:
    result = await run_case(_case("pii-001"))
    assert {"answer", "result", "spans", "audit"} <= set(result.outputs)
    assert any(k.startswith("file:checkpoints") for k in result.outputs)
    assert any(k.startswith("file:memory") for k in result.outputs)
    report = compute_metrics([result], corpus_keys=set())
    assert report.pii_leak_count == 0


async def test_report_and_gate_on_a_small_suite(tmp_path: Path) -> None:
    cases = [_case(i) for i in ("norm-001", "au-h07", "bl-h03", "edge-006")]
    results = await run_suite(cases)
    keys = corpus_keys(cases)
    flat = compute_metrics(results, corpus_keys=keys).flat()
    gate = evaluate_gate(flat, tmp_path / "baseline.json")
    assert not gate.passed  # no baseline exists: never silently accepted
    meta = {
        "model": "mock",
        "dataset_version": "test",
        "cases": len(results),
        "started_at": "now",
        "judge": False,
    }
    report = build_report(results, corpus_keys=keys, gate=gate, meta=meta)
    markdown = render_markdown(report, gate)
    assert "These are not real-model results" in markdown
    assert "| all | dev | heldout |" in markdown
    assert "CI" in markdown
    assert "needs judge" in markdown  # edge-006 is judge-only
    report.pop("_metrics_objects")
    serialized = json.dumps(report)
    assert not any(c.question in serialized for c in cases)  # ids and reasons only


def test_cli_writes_reports_and_fails_without_a_baseline(tmp_path: Path) -> None:
    args = ["--case", "norm-002", "--out", str(tmp_path), "--baseline", str(tmp_path / "b.json")]
    assert eval_run.main(args) == 1
    assert (tmp_path / "report.md").exists()
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert data["meta"]["model"] == "mock"
    assert data["gate"]["passed"] is False
    assert eval_run.main([*args, "--report-only"]) == 0
    assert not (tmp_path / "b.json").exists()  # the run never writes a baseline
