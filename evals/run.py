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
from evals.harness import HarnessConfig, SuiteStoppedError, corpus_keys, run_suite
from evals.judge import CachedJudge, LLMJudge
from evals.metrics import compute_metrics
from evals.offline import block_network
from evals.report import build_report, render_markdown
from evals.resultsio import write_results
from evals.runlog import RunInProgressError, RunLog, exclusive_run, new_run_log
from evals.schema import CaseResult, EvalCase
from evals.subsets import FAST_CASES
from sentinel.llm_ratelimit import ProviderUsage

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


def _config(llm: str, case_delay_s: float = 0.0, token_budget: int | None = None) -> HarnessConfig:
    from sentinel.config import Settings, get_settings

    settings = Settings() if llm == "mock" else get_settings()
    if llm == "mock":
        return HarnessConfig(
            case_delay_s=case_delay_s,
            token_budget=token_budget,
            injection_guard=settings.injection_guard,
        )
    from sentinel.graph.nodes.planner import make_planner
    from sentinel.llm import make_chat_model

    if settings.llm_provider == "mock":
        raise SystemExit("--llm configured needs LLM_PROVIDER/LLM_MODEL/LLM_API_KEY set")
    usage = ProviderUsage()
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    return HarnessConfig(
        llm_factory=lambda: make_chat_model(settings, usage=usage),
        planner_factory=lambda model: make_planner(settings.llm_provider, model),
        tokens_reported=True,
        model_name=f"{settings.llm_provider}:{settings.llm_model}",
        tool_timeout_s=settings.tool_timeout_s,
        usage=usage,
        case_delay_s=case_delay_s,
        token_budget=token_budget,
        secrets=(key,) if key else (),
        injection_guard=settings.injection_guard,
    )


def _judge(config: HarnessConfig, cache: Path) -> CachedJudge:
    """The judge uses LLM_JUDGE_MODEL when set (same provider and key), else the agent model.
    Its verdict cache is keyed by the judge model."""
    from sentinel.config import get_settings
    from sentinel.llm import make_chat_model

    settings = get_settings()
    model = settings.llm_judge_model or settings.llm_model
    llm = make_chat_model(settings, model=model, usage=config.usage)
    return CachedJudge(LLMJudge(llm), cache, model=f"{settings.llm_provider}:{model}")


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
        "--case-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Pause between cases (stays under per-minute rate limits; not counted as latency).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Stop before a case that would likely push the run's provider tokens past this; "
        "a partial report is still written (exit 1).",
    )
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
    if args.case_delay < 0:
        raise SystemExit("--case-delay must be 0 or more")
    if args.token_budget is not None and args.token_budget <= 0:
        raise SystemExit("--token-budget must be a positive number of tokens")
    config = _config(args.llm, args.case_delay, args.token_budget)
    subset = "filtered" if args.case or args.category else ("fast" if args.fast else "all")
    started = datetime.now(UTC)
    try:
        with exclusive_run(args.out):
            rid, log = new_run_log(args.out, started, config.model_name, subset)
            with log:
                return _run(args, cases, config, subset, started, rid, log)
    except RunInProgressError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


def _progress_line(i: int, total: int, result: CaseResult) -> str:
    wait = f" (provider wait {result.provider_wait_ms} ms)" if result.provider_wait_ms else ""
    if result.harness_error:
        return (
            f"[{i:>3}/{total}] {result.case_id:<10} error: "
            + (result.harness_error_detail or result.harness_error)
            + wait
        )
    return f"[{i:>3}/{total}] {result.case_id:<10} ok{wait}"


def _run(
    args: argparse.Namespace,
    cases: list[EvalCase],
    config: HarnessConfig,
    subset: str,
    started: datetime,
    rid: str,
    log: RunLog,
) -> int:
    log.line(f"run {rid}: {len(cases)} cases, model {config.model_name}, log {log.path}")
    if config.usage is not None:
        log.line(f"warning: calling a real model ({config.model_name})")
    if config.token_budget is not None:
        log.line(f"token budget: {config.token_budget}")
    judge = _judge(config, args.judge_cache) if args.judge else None

    def progress(i: int, total: int, result: CaseResult) -> None:
        log.line(_progress_line(i, total, result))

    offline = block_network() if args.llm == "mock" else nullcontext()
    partial: dict[str, object] | None = None
    try:
        with offline:
            results = anyio.run(run_suite, cases, config, progress)
    except SuiteStoppedError as exc:
        results = exc.results
        partial = {"reason": exc.reason, "completed": len(results), "planned": exc.planned}
        log.line(f"stopped: {exc}", err=True)
        if judge is not None:
            log.line("judge skipped: the run stopped early", err=True)
            judge = None
    if not results:
        log.line("no case finished, so no report was written", err=True)
        return 1
    keys = corpus_keys(cases)
    flat = compute_metrics(results, corpus_keys=keys, k=args.k, judge=judge).flat()
    baseline_path = args.baseline or default_baseline(config.model_name, subset)
    gate = evaluate_gate(flat, baseline_path)
    note = baseline_note(gate, advisory=args.baseline_advisory)
    usage = config.usage
    meta = {
        "model": config.model_name,
        "subset": subset,
        "baseline": baseline_path.as_posix(),
        "dataset_version": read_manifest(args.datasets).version,
        "dataset_files": file_hashes(args.datasets),
        "cases": len(results),
        "commit": _commit(),
        "started_at": started.isoformat(timespec="seconds"),
        "run_id": rid,
        "log": log.path.as_posix(),
        "partial": partial,
        "judge": judge is not None,
        "judge_cache": {"hits": judge.hits, "misses": judge.misses} if judge else None,
        "judge_model": judge.model if judge else None,
        "k": args.k,
        "baseline_note": note,
        "case_delay_s": config.case_delay_s,
        "token_budget": config.token_budget,
        "provider": (
            {
                "rate_limit_retries": usage.rate_limit_retries,
                "server_retries": usage.server_retries,
                "waited_s": round(usage.waited_s, 1),
                "tokens": usage.tokens,
            }
            if usage
            else None
        ),
    }
    report = build_report(results, corpus_keys=keys, gate=gate, meta=meta, k=args.k, judge=judge)
    args.out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report, gate)
    report.pop("_metrics_objects")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    results_path = write_results(args.out, results)
    log.line(f"wrote {args.out / 'report.md'} and {args.out / 'metrics.json'}")
    log.line(f"wrote {results_path} ({len(results)} case results for merge)")
    if usage is not None:
        log.line(
            f"provider: {usage.tokens} tokens, {usage.rate_limit_retries} rate-limit and "
            f"{usage.server_retries} server retries, {usage.waited_s:.1f}s waiting"
        )
    log.line(f"gate: {'PASS' if gate.passed else 'FAIL'}")
    if note:
        log.line(note)
    if partial is not None:
        log.line(f"partial run: {partial['completed']} of {partial['planned']} cases", err=True)
        return 0 if args.report_only else 1
    if args.report_only:
        return 0
    return exit_code(gate, results, baseline_advisory=args.baseline_advisory)


if __name__ == "__main__":
    sys.exit(main())
