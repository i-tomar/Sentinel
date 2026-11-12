"""Run the eval suite: `uv run python -m evals.run` (or `make eval`).

Default: every dataset case through the real agent graph on in-process fixture servers with
the deterministic MockLLM, with public network connections refused. No credentials, no
baseline written. `--fast` runs the eval-fast subset (`make eval-fast`, every-PR CI). A real
model is opt-in with `--llm configured` (reads LLM_PROVIDER / LLM_MODEL / LLM_API_KEY; the
optional eval.yml workflow), plus `--judge` for the cached LLM judge.

Exit code 1 when a case cannot run or the gate fails (including "no baseline"). Two escape
hatches: `--report-only` (always 0) and `--baseline-advisory` (a baseline problem alone does
not fail). The every-PR job no longer uses that flag. The real-model workflow still does,
because no real-model baseline is approved.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import anyio

from evals.datasets import DATASETS_DIR, file_hashes, load_cases, read_manifest
from evals.gate import GateResult, evaluate_gate
from evals.harness import HarnessConfig, corpus_keys, run_suite
from evals.judge import CachedJudge, LLMJudge
from evals.metrics import compute_metrics
from evals.offline import block_network
from evals.report import build_report, render_markdown
from evals.schema import CaseResult
from evals.subsets import FAST_CASES

BASELINES_DIR = Path(__file__).parent / "baselines"


def default_baseline(model: str, subset: str) -> Path:
    """One baseline per (model, subset): fast-subset and full-suite numbers, and mock and
    real-model numbers, are not comparable with each other."""
    slug = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in model.lower())
    return BASELINES_DIR / f"{slug}-{subset}.json"


def _commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _config(llm: str) -> HarnessConfig:
    if llm == "mock":
        return HarnessConfig()
    from sentinel.config import get_settings
    from sentinel.graph.nodes.planner import make_planner
    from sentinel.llm import make_chat_model

    settings = get_settings()
    if settings.llm_provider == "mock":
        raise SystemExit("--llm configured needs LLM_PROVIDER/LLM_MODEL/LLM_API_KEY set")
    print(f"warning: calling a real model ({settings.llm_provider}:{settings.llm_model})")
    return HarnessConfig(
        llm_factory=lambda: make_chat_model(settings),
        planner_factory=lambda model: make_planner(settings.llm_provider, model),
        tokens_reported=True,
        model_name=f"{settings.llm_provider}:{settings.llm_model}",
        tool_timeout_s=settings.tool_timeout_s,
    )


def baseline_note(gate: GateResult, *, advisory: bool) -> str | None:
    """A loud, human-readable line when the baseline check failed (missing, unreadable,
    unapproved or edited); None when the baseline is fine."""
    check = next((c for c in gate.checks if c.metric == "baseline"), None)
    if check is None or check.status != "fail":
        return None
    if advisory:
        return (
            f"Baseline problem: {check.reason}. The gate reports this as a failure, but "
            "--baseline-advisory is set, so a baseline problem alone does not fail this run. "
            "Hard rules (unapproved writes, PII leaks, hallucinated citations) still fail it."
        )
    return f"Baseline problem: {check.reason}. This fails the gate."


def exit_code(gate: GateResult, results: Sequence[CaseResult], *, baseline_advisory: bool) -> int:
    """1 if any case could not run, or any gate check failed. With `baseline_advisory` a
    failing *baseline* check alone does not cause exit 1; every other failure, hard rules
    first, still does. The gate result itself is never changed."""
    if any(r.harness_error for r in results):
        return 1
    failed = [c for c in gate.checks if c.status == "fail"]
    if baseline_advisory:
        failed = [c for c in failed if c.metric != "baseline"]
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS_DIR)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Default: evals/baselines/<model>-<fast|all|filtered>.json",
    )
    parser.add_argument("--case", action="append", default=[], help="Only these case ids.")
    parser.add_argument("--category", action="append", default=[], help="Only these categories.")
    parser.add_argument("--fast", action="store_true", help="Only the eval-fast subset.")
    parser.add_argument("--list", action="store_true", help="Print the selected case ids, exit.")
    parser.add_argument("--llm", choices=["mock", "configured"], default="mock")
    parser.add_argument("--judge", action="store_true", help="LLM judge (needs --llm configured).")
    parser.add_argument("--judge-cache", type=Path, default=Path(".cache/eval-judge"))
    parser.add_argument("--k", type=int, default=5, help="k for recall@k.")
    parser.add_argument("--report-only", action="store_true", help="Exit 0 even if the gate fails.")
    parser.add_argument(
        "--baseline-advisory",
        action="store_true",
        help="A missing/invalid baseline is reported but does not fail the exit code. "
        "Hard rules and other gate failures still do. The real-model workflow uses this "
        "until a baseline for that model is approved.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("presidio-analyzer", "mcp", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cases = load_cases(args.datasets)
    if args.fast:
        cases = [c for c in cases if c.id in set(FAST_CASES)]
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if args.category:
        cases = [c for c in cases if c.category in set(args.category)]
    if args.list:
        print("\n".join(c.id for c in cases))
        return 0
    if args.judge and args.llm != "configured":
        raise SystemExit("--judge needs --llm configured (the mock cannot judge)")
    config = _config(args.llm)
    judge = (
        CachedJudge(LLMJudge(config.llm_factory()), args.judge_cache, model=config.model_name)
        if args.judge
        else None
    )
    started = datetime.now(UTC)

    def progress(i: int, total: int, result: CaseResult) -> None:
        status = "error" if result.harness_error else "ok"
        print(f"[{i:>3}/{total}] {result.case_id:<10} {status}", flush=True)

    offline = block_network() if args.llm == "mock" else nullcontext()
    with offline:
        results = anyio.run(run_suite, cases, config, progress)
    keys = corpus_keys(cases)
    flat = compute_metrics(results, corpus_keys=keys, k=args.k, judge=judge).flat()
    subset = "filtered" if args.case or args.category else ("fast" if args.fast else "all")
    baseline_path = args.baseline or default_baseline(config.model_name, subset)
    gate = evaluate_gate(flat, baseline_path)
    note = baseline_note(gate, advisory=args.baseline_advisory)
    meta = {
        "model": config.model_name,
        "subset": subset,
        "baseline": baseline_path.as_posix(),
        "dataset_version": read_manifest(args.datasets).version,
        "dataset_files": file_hashes(args.datasets),
        "cases": len(results),
        "commit": _commit(),
        "started_at": started.isoformat(timespec="seconds"),
        "judge": judge is not None,
        "judge_cache": {"hits": judge.hits, "misses": judge.misses} if judge else None,
        "k": args.k,
        "baseline_note": note,
    }
    report = build_report(results, corpus_keys=keys, gate=gate, meta=meta, k=args.k, judge=judge)
    args.out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report, gate)
    report.pop("_metrics_objects")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out / 'report.md'} and {args.out / 'metrics.json'}")
    print(f"gate: {'PASS' if gate.passed else 'FAIL'}")
    if note:
        print(note)
    if args.report_only:
        return 0
    return exit_code(gate, results, baseline_advisory=args.baseline_advisory)


if __name__ == "__main__":
    sys.exit(main())
