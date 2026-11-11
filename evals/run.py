"""Run the eval suite: `uv run python -m evals.run` (or `make eval`).

Default: every dataset case through the real agent graph on in-process fixture servers with
the deterministic MockLLM. No network, no credentials, no baseline written. A real model is
opt-in with `--llm configured` (reads LLM_PROVIDER / LLM_MODEL / LLM_API_KEY) and is never
used by tests or CI by default. Exit code 1 when the gate fails (including "no baseline"),
unless `--report-only`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import anyio

from evals.datasets import DATASETS_DIR, file_hashes, load_cases, read_manifest
from evals.gate import evaluate_gate
from evals.harness import HarnessConfig, corpus_keys, run_suite
from evals.metrics import compute_metrics
from evals.report import build_report, render_markdown
from evals.schema import CaseResult

BASELINE = Path(__file__).parent / "baseline.json"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--datasets", type=Path, default=DATASETS_DIR)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--case", action="append", default=[], help="Only these case ids.")
    parser.add_argument("--category", action="append", default=[], help="Only these categories.")
    parser.add_argument("--llm", choices=["mock", "configured"], default="mock")
    parser.add_argument("--k", type=int, default=5, help="k for recall@k.")
    parser.add_argument("--report-only", action="store_true", help="Exit 0 even if the gate fails.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("presidio-analyzer", "mcp", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cases = load_cases(args.datasets)
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if args.category:
        cases = [c for c in cases if c.category in set(args.category)]
    config = _config(args.llm)
    started = datetime.now(UTC)

    def progress(i: int, total: int, result: CaseResult) -> None:
        status = "error" if result.harness_error else "ok"
        print(f"[{i:>3}/{total}] {result.case_id:<10} {status}", flush=True)

    results = anyio.run(run_suite, cases, config, progress)
    keys = corpus_keys(cases)
    flat = compute_metrics(results, corpus_keys=keys, k=args.k).flat()
    gate = evaluate_gate(flat, args.baseline)
    meta = {
        "model": config.model_name,
        "dataset_version": read_manifest(args.datasets).version,
        "dataset_files": file_hashes(args.datasets),
        "cases": len(results),
        "commit": _commit(),
        "started_at": started.isoformat(timespec="seconds"),
        "judge": False,
        "k": args.k,
    }
    report = build_report(results, corpus_keys=keys, gate=gate, meta=meta, k=args.k)
    args.out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report, gate)
    report.pop("_metrics_objects")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out / 'report.md'} and {args.out / 'metrics.json'}")
    print(f"gate: {'PASS' if gate.passed else 'FAIL'}")
    return 0 if gate.passed or args.report_only else 1


if __name__ == "__main__":
    sys.exit(main())
