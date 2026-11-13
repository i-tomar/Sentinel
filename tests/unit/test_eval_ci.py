"""CI wiring for the evals (step q): exit policy, red-gate proof, fast subset, offline guard,
judge cache and the workflow files themselves. Nothing here runs a model or the network."""

from __future__ import annotations

import socket
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from evals.datasets import load_cases
from evals.gate import approve_baseline, evaluate_gate
from evals.judge import CachedJudge
from evals.offline import NetworkBlocked, block_network
from evals.run import baseline_note, exit_code, main
from evals.schema import CaseResult
from evals.subsets import FAST_CASES

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

GOOD: dict[str, float | int | None] = {
    "answer_correct": 0.60,
    "answer_correct.n": 74,
    "citation_precision": 0.70,
    "citation_precision.n": 353,
    "citation_hallucinated": 0.0,
    "citation_hallucinated.n": 105,
    "recall_at_k": 0.85,
    "recall_at_k.n": 80,
    "block_rate_input": 1.0,
    "block_rate_input.n": 23,
    "block_rate_retrieved": 0.65,
    "block_rate_retrieved.n": 20,
    "false_positive_rate_input": 0.023,
    "false_positive_rate_input.n": 44,
    "false_positive_rate_retrieved": 0.10,
    "false_positive_rate_retrieved.n": 10,
    "unapproved_writes": 0,
    "pii_leak_count": 0,
    "latency_p95_ms": 700,
    "latency_p95_ms.n": 124,
}


def _ok(case_id: str = "c1", **fields: object) -> CaseResult:
    return CaseResult.model_validate(
        {"case_id": case_id, "category": "normal", "answer": "x", **fields}
    )


# --- prove the gate can go red ------------------------------------------------------------


@pytest.mark.parametrize(
    ("degradation", "failing"),
    [
        ({"answer_correct": 0.55}, "answer_correct"),  # -5 pp, tolerance 2 pp
        # what running without retrieved_guard would look like: attacks get through
        (
            {"block_rate_retrieved": 0.40, "false_positive_rate_retrieved": 0.0},
            "block_rate_retrieved",
        ),
        ({"pii_leak_count": 3}, "pii_leak_count"),
        ({"unapproved_writes": 1}, "unapproved_writes"),
        ({"citation_hallucinated": 0.02}, "citation_hallucinated"),
    ],
)
def test_a_degraded_run_turns_the_gate_red(
    tmp_path: Path, degradation: dict[str, float], failing: str
) -> None:
    baseline = tmp_path / "baseline.json"
    approve_baseline(GOOD, baseline, approved_by="test-owner")
    assert evaluate_gate(GOOD, baseline).passed  # the healthy run is green
    gate = evaluate_gate({**GOOD, **degradation}, baseline)
    assert not gate.passed
    assert next(c for c in gate.checks if c.metric == failing).status == "fail"
    assert exit_code(gate, [_ok()], baseline_advisory=True) == 1  # fails CI even in advisory


# --- the advisory flag (real-model workflow only; the every-PR job no longer uses it) ------


def test_missing_baseline_is_a_gate_failure_but_advisory_in_ci(tmp_path: Path) -> None:
    gate = evaluate_gate(GOOD, tmp_path / "baseline.json")
    assert not gate.passed  # the library is unchanged: no baseline means the gate fails
    assert exit_code(gate, [_ok()], baseline_advisory=False) == 1  # make eval / make eval-fast
    assert exit_code(gate, [_ok()], baseline_advisory=True) == 0  # the required PR check
    note = baseline_note(gate, advisory=True)
    assert note is not None
    assert "no baseline" in note
    assert "baseline problem alone" in note
    assert baseline_note(gate, advisory=False) is not None


def test_a_hand_edited_baseline_is_reported_and_advisory(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    approve_baseline(GOOD, baseline, approved_by="test-owner")
    baseline.write_text(baseline.read_text().replace("0.85", "0.10"), encoding="utf-8")
    gate = evaluate_gate(GOOD, baseline)
    note = baseline_note(gate, advisory=True)
    assert note is not None
    assert "changed" in note
    assert exit_code(gate, [_ok()], baseline_advisory=True) == 0


def test_hard_rules_fail_even_without_a_baseline(tmp_path: Path) -> None:
    gate = evaluate_gate({**GOOD, "pii_leak_count": 1}, tmp_path / "missing.json")
    assert exit_code(gate, [_ok()], baseline_advisory=True) == 1


def test_a_case_that_could_not_run_fails_the_job(tmp_path: Path) -> None:
    gate = evaluate_gate(GOOD, tmp_path / "missing.json")
    broken = _ok(harness_error="RuntimeError")
    assert exit_code(gate, [_ok(), broken], baseline_advisory=True) == 1


def test_a_healthy_run_needs_no_note(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    approve_baseline(GOOD, baseline, approved_by="test-owner")
    gate = evaluate_gate(GOOD, baseline)
    assert baseline_note(gate, advisory=True) is None
    assert exit_code(gate, [_ok()], baseline_advisory=False) == 0


# --- the fast subset ----------------------------------------------------------------------


def test_the_fast_subset_exists_and_is_representative(capsys: pytest.CaptureFixture[str]) -> None:
    cases = {c.id: c for c in load_cases()}
    assert len(set(FAST_CASES)) == len(FAST_CASES)
    assert set(FAST_CASES) <= set(cases)
    chosen = [cases[i] for i in FAST_CASES]
    categories = Counter(c.category for c in chosen)
    assert set(categories) == {c.category for c in cases.values()}  # every category
    assert {c.split for c in chosen if c.category.startswith("adversarial")} == {"dev", "heldout"}
    tags = {t for c in chosen for t in c.tags}
    assert {"write-request", "outside-domain", "pii", "multi-hop", "stale-vs-current"} <= tags
    assert {c.fixture_overrides for c in chosen if c.fixture_overrides} >= {
        "gmail=timeout",
        "*=error_500",
        "notion.get_page=malformed",
    }
    assert len(FAST_CASES) < len(cases)
    assert main(["--fast", "--list"]) == 0
    assert sorted(capsys.readouterr().out.split()) == sorted(FAST_CASES)


# --- no public network during evals -------------------------------------------------------


def test_block_network_refuses_public_hosts_and_restores() -> None:
    with block_network(), pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with block_network():
        socket.getaddrinfo("127.0.0.1", 80)  # loopback stays allowed


# --- the judge cache ----------------------------------------------------------------------


class CountingJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, question: str, answer: str, rubric: str) -> bool | None:
        self.calls += 1
        return "not" in answer


def test_judge_results_are_cached_by_hash(tmp_path: Path) -> None:
    inner = CountingJudge()
    first = CachedJudge(inner, tmp_path, model="m1")
    assert first.judge("Is it done?", "It is not done [1].", "Must say no.") is True
    assert first.judge("Is it done?", "It is not done [1].", "Must say no.") is True
    assert inner.calls == 1
    assert (first.hits, first.misses) == (1, 1)
    again = CachedJudge(inner, tmp_path, model="m1")  # a later run, same cache directory
    assert again.judge("Is it done?", "It is not done [1].", "Must say no.") is True
    assert inner.calls == 1
    other_model = CachedJudge(inner, tmp_path, model="m2")
    other_model.judge("Is it done?", "It is not done [1].", "Must say no.")
    assert inner.calls == 2  # the model is part of the key
    stored = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json"))
    assert "Is it done?" not in stored  # cache files hold verdicts only, never text


# --- the workflow files -------------------------------------------------------------------


def _workflow(name: str) -> dict[Any, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _steps(job: dict[Any, Any]) -> str:
    return str(yaml.safe_dump(job.get("steps")))


def test_ci_runs_eval_fast_with_no_secrets() -> None:
    raw = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    ci = _workflow("ci.yml")
    jobs = ci["jobs"]
    assert isinstance(jobs, dict)
    assert {"lint", "test", "eval-fast"} <= set(jobs)
    steps = _steps(jobs["eval-fast"])
    assert "make eval-fast" in steps
    assert "--baseline-advisory" not in steps
    assert "GITHUB_STEP_SUMMARY" in steps
    assert "actions/upload-artifact" in steps
    assert "secrets." not in raw
    assert "LLM_PROVIDER: mock" in raw


def test_full_eval_workflow_triggers_and_skips_without_a_secret() -> None:
    raw = (WORKFLOWS / "eval.yml").read_text(encoding="utf-8")
    wf = _workflow("eval.yml")
    triggers = wf.get("on", wf.get(True))  # YAML 1.1 reads a bare `on` key as True
    assert isinstance(triggers, dict)
    assert {"pull_request", "schedule", "workflow_dispatch"} <= set(triggers)
    assert "run-evals" in raw
    assert "secrets.LLM_API_KEY" in raw
    assert "LLM_API_KEY secret is not configured" in raw  # clean skip, reported
    assert "actions/cache" in raw  # judge results cached by hash
    assert "GITHUB_STEP_SUMMARY" in raw
    assert "actions/upload-artifact" in raw
    assert "--baseline-advisory" in raw  # no real-model baseline is approved yet


def test_full_eval_workflow_supports_groq_with_the_same_skip() -> None:
    raw = (WORKFLOWS / "eval.yml").read_text(encoding="utf-8")
    job = _workflow("eval.yml")["jobs"]["eval"]
    env = job["env"]
    assert env["LLM_API_KEY"] == "${{ secrets.LLM_API_KEY }}"  # one secret for every provider
    assert "vars.LLM_PROVIDER" in env["LLM_PROVIDER"]  # set it to groq
    assert env["LLM_JUDGE_MODEL"] == "${{ vars.LLM_JUDGE_MODEL }}"
    assert "vars.EVAL_CASE_DELAY_S" in env["EVAL_CASE_DELAY_S"]
    assert '--case-delay "${EVAL_CASE_DELAY_S}"' in raw  # via env, not template-expanded
    assert "LLM_JUDGE_MODEL || env.LLM_MODEL" in raw  # judge cache keyed by the judge model
    assert job["timeout-minutes"] >= 120
    first = job["steps"][0]
    assert first["id"] == "secret"
    assert 'if [ -z "${LLM_API_KEY}" ]' in first["run"]  # skip without the secret, any provider
    assert "groq" in raw
