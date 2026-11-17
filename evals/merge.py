"""Merge partial eval report directories into one suite report.

When every part has `results.jsonl`, CaseResults are concatenated (dataset order) and every
metric is recomputed with `compute_metrics`. Older parts that only wrote `metrics.json` still
merge: rates and counts are rebuilt by pooling each part's sufficient statistics
(successes/n, weighted means) — the same numbers concatenating the original CaseResults
would have produced — not by averaging percentages. Latency percentiles cannot be recovered
without per-case latencies and are left unset.

Usage:
  uv run python -m evals.merge reports/part-a reports/part-b --out reports/merged \\
      --require-commit 7803d20 --subset all --baseline-advisory
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.datasets import DATASETS_DIR, load_cases
from evals.gate import evaluate_gate
from evals.harness import corpus_keys
from evals.metrics import (
    CitationScores,
    LatencyAndTools,
    Mean,
    Metrics,
    PermissionScores,
    PiiLeaks,
    Rate,
    TokenUsage,
    compute_metrics,
    rate,
)
from evals.report import build_report, render_markdown
from evals.resultsio import has_results, load_results, ordered_by_ids, write_results
from evals.run import baseline_note, default_baseline
from evals.schema import CaseResult

METRICS_NAME = "metrics.json"
SPLITS = ("all", "dev", "heldout")


class MergeError(Exception):
    pass


def _load_metrics(part: Path) -> dict[str, Any]:
    path = part / METRICS_NAME
    if not path.is_file():
        raise MergeError(f"no {METRICS_NAME} in {part}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MergeError(f"unreadable {path}: {exc}") from None
    if not isinstance(data, dict):
        raise MergeError(f"{path} is not a JSON object")
    return data


def _require_same(parts: Sequence[dict[str, Any]], key: str, *, label: str) -> Any:
    values = [p["meta"].get(key) for p in parts]
    if any(v != values[0] for v in values[1:]):
        raise MergeError(f"{label} differ across parts: {values!r}")
    return values[0]


def validate_parts(
    parts: Sequence[dict[str, Any]],
    *,
    require_commit: str,
    planned_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Fail if commits, dataset hashes, or case coverage do not line up.

    Returns case_id -> summary row from the parts (id, category, split, surface, failures).
    """
    if len(parts) < 2:
        raise MergeError("need at least two parts to merge")
    commit = _require_same(parts, "commit", label="commits")
    if commit != require_commit:
        raise MergeError(f"commit is {commit!r}, expected {require_commit!r}")
    _require_same(parts, "dataset_files", label="dataset file hashes")
    _require_same(parts, "dataset_version", label="dataset versions")
    _require_same(parts, "model", label="models")
    _require_same(parts, "k", label="k")

    by_id: dict[str, dict[str, Any]] = {}
    owner: dict[str, int] = {}
    for index, part in enumerate(parts):
        for case in part["cases"]:
            cid = case["id"]
            if cid in owner:
                raise MergeError(f"case {cid!r} appears in part {owner[cid]} and part {index}")
            owner[cid] = index
            by_id[cid] = case
    planned = list(planned_ids)
    missing = [i for i in planned if i not in by_id]
    extra = sorted(set(by_id) - set(planned))
    if missing or extra:
        raise MergeError(
            f"merged cases must be exactly the planned suite: missing={missing!r} extra={extra!r}"
        )
    if len(by_id) != len(planned):
        raise MergeError(f"expected {len(planned)} cases, got {len(by_id)}")
    return by_id


def _sum_provider(parts: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    providers = [p["meta"].get("provider") for p in parts]
    if all(p is None for p in providers):
        return None
    if any(p is None for p in providers):
        raise MergeError("provider usage present in some parts but not others")
    rows: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            raise MergeError("provider usage must be an object")
        rows.append(provider)
    return {
        "rate_limit_retries": sum(int(p["rate_limit_retries"]) for p in rows),
        "server_retries": sum(int(p["server_retries"]) for p in rows),
        "waited_s": round(sum(float(p["waited_s"]) for p in rows), 1),
        "tokens": sum(int(p["tokens"]) for p in rows),
    }


def _pool_rate(rates: Sequence[Rate]) -> Rate:
    return rate(sum(r.successes for r in rates), sum(r.n for r in rates))


def _pool_mean(means: Sequence[Mean]) -> Mean:
    n = sum(m.n for m in means)
    if n == 0 or any(m.value is None for m in means if m.n):
        return Mean(n=n, value=None, stderr=None)
    value = sum((m.value or 0.0) * m.n for m in means) / n
    return Mean(n=n, value=value, stderr=None)


def _pool_tokens(items: Sequence[TokenUsage | None]) -> TokenUsage | None:
    # Empty splits legitimately report tokens=None; ignore those and pool the rest.
    present = [t for t in items if t is not None]
    if not present:
        return None
    n = sum(t.reported_cases for t in present)
    if n == 0:
        return None
    return TokenUsage(
        reported_cases=n,
        mean_input=sum(t.mean_input * t.reported_cases for t in present) / n,
        mean_output=sum(t.mean_output * t.reported_cases for t in present) / n,
    )


def _pool_citations(items: Sequence[CitationScores]) -> CitationScores:
    total = sum(c.total for c in items)
    return CitationScores(
        total=total,
        existing=_pool_rate([c.existing for c in items]),
        returned=_pool_rate([c.returned for c in items]),
        supported=_pool_rate([c.supported for c in items]),
        unverifiable=sum(c.unverifiable for c in items),
        precision=_pool_rate([c.precision for c in items]),
        recall=_pool_rate([c.recall for c in items]),
        forbidden_hits=sum(c.forbidden_hits for c in items),
        hallucinated_answers=_pool_rate([c.hallucinated_answers for c in items]),
    )


def _pool_permissions(items: Sequence[PermissionScores]) -> PermissionScores:
    return PermissionScores(
        unapproved_writes=sum(p.unapproved_writes for p in items),
        approval_requested=_pool_rate([p.approval_requested for p in items]),
        correct_refusals=_pool_rate([p.correct_refusals for p in items]),
        over_refusals=_pool_rate([p.over_refusals for p in items]),
        policy_denials=_pool_rate([p.policy_denials for p in items]),
    )


def _pool_pii(items: Sequence[PiiLeaks]) -> PiiLeaks:
    by_path: dict[str, int] = {}
    cases: list[str] = []
    count = 0
    for p in items:
        count += p.count
        cases.extend(p.cases)
        for key, value in p.by_path.items():
            by_path[key] = by_path.get(key, 0) + value
    return PiiLeaks(count=count, by_path=by_path, cases=cases)


def _pool_latency(items: Sequence[LatencyAndTools]) -> LatencyAndTools:
    """Percentiles need per-case latencies; only n and mean tool calls pool exactly."""
    n = sum(x.n for x in items)
    if n == 0:
        return LatencyAndTools(n=0, p50_ms=None, p95_ms=None, mean_tool_calls=None)
    mean_tools = sum((x.mean_tool_calls or 0.0) * x.n for x in items) / n
    return LatencyAndTools(n=n, p50_ms=None, p95_ms=None, mean_tool_calls=mean_tools)


def pool_metrics(items: Sequence[Metrics]) -> Metrics:
    """Rebuild Metrics by pooling sufficient statistics from disjoint partitions."""
    if not items:
        raise MergeError("no metrics to pool")
    judged = [m.answer_judged for m in items]
    answer_judged = (
        None if all(j is None for j in judged) else _pool_rate([j for j in judged if j is not None])
    )
    return Metrics(
        cases=sum(m.cases for m in items),
        answer_correct=_pool_rate([m.answer_correct for m in items]),
        answer_judged=answer_judged,
        citations=_pool_citations([m.citations for m in items]),
        recall_at_k=_pool_mean([m.recall_at_k for m in items]),
        k=items[0].k,
        block_rate_input=_pool_rate([m.block_rate_input for m in items]),
        block_rate_retrieved=_pool_rate([m.block_rate_retrieved for m in items]),
        false_positive_rate_input=_pool_rate([m.false_positive_rate_input for m in items]),
        false_positive_rate_retrieved=_pool_rate([m.false_positive_rate_retrieved for m in items]),
        permissions=_pool_permissions([m.permissions for m in items]),
        unapproved_writes=sum(m.unapproved_writes for m in items),
        pii=_pool_pii([m.pii for m in items]),
        pii_leak_count=sum(m.pii_leak_count for m in items),
        latency=_pool_latency([m.latency for m in items]),
        tokens=_pool_tokens([m.tokens for m in items]),
        needs_judge_unscored=sum(m.needs_judge_unscored for m in items),
        input_attacks_refused=_pool_rate([m.input_attacks_refused for m in items]),
        retrieved_attacks_flagged=_pool_rate([m.retrieved_attacks_flagged for m in items]),
    )


def _merged_meta(
    parts: Sequence[dict[str, Any]],
    *,
    cases: int,
    subset: str,
    baseline: Path,
    baseline_note: str | None,
    source: str,
) -> dict[str, Any]:
    first = parts[0]["meta"]
    part_rows = []
    for part in parts:
        meta = part["meta"]
        provider = meta.get("provider") or {}
        part_rows.append(
            {
                "run_id": meta.get("run_id"),
                "started_at": meta.get("started_at"),
                "cases": meta.get("cases"),
                "tokens": provider.get("tokens"),
                "log": meta.get("log"),
            }
        )
    started = min(str(p["meta"].get("started_at") or "") for p in parts)
    run_ids = [str(p["meta"].get("run_id") or "") for p in parts]
    return {
        "model": first["model"],
        "subset": subset,
        "baseline": baseline.as_posix(),
        "dataset_version": first["dataset_version"],
        "dataset_files": first["dataset_files"],
        "cases": cases,
        "commit": first["commit"],
        "started_at": started,
        "run_id": "merged:" + "+".join(run_ids),
        "log": None,
        "partial": None,
        "judge": any(bool(p["meta"].get("judge")) for p in parts),
        "judge_cache": None,
        "judge_model": first.get("judge_model"),
        "k": first.get("k", 5),
        "baseline_note": baseline_note,
        "case_delay_s": None,
        "token_budget": None,
        "provider": _sum_provider(parts),
        "merged": {
            "note": "run in 2 parts on different days",
            "parts": part_rows,
            "source": source,
        },
    }


def _write_report(out: Path, report: dict[str, Any], gate: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report, gate)
    report.pop("_metrics_objects", None)
    (out / METRICS_NAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown, encoding="utf-8")


def merge_from_results(
    part_dirs: Sequence[Path],
    metrics_parts: Sequence[dict[str, Any]],
    *,
    planned_ids: Sequence[str],
    cases_for_keys: Sequence[Any],
    out: Path,
    subset: str,
    baseline_path: Path,
    baseline_advisory: bool,
    k: int,
) -> dict[str, Any]:
    combined: list[CaseResult] = []
    for part_dir in part_dirs:
        combined.extend(load_results(part_dir))
    ordered = ordered_by_ids(combined, planned_ids)
    keys = corpus_keys(cases_for_keys)
    metrics_by_split = {
        "all": compute_metrics(ordered, corpus_keys=keys, k=k, judge=None),
        "dev": compute_metrics(
            [r for r in ordered if r.split == "dev"], corpus_keys=keys, k=k, judge=None
        ),
        "heldout": compute_metrics(
            [r for r in ordered if r.split == "heldout"], corpus_keys=keys, k=k, judge=None
        ),
    }
    flat = metrics_by_split["all"].flat()
    gate = evaluate_gate(flat, baseline_path)
    note = baseline_note(gate, advisory=baseline_advisory)
    meta = _merged_meta(
        metrics_parts,
        cases=len(ordered),
        subset=subset,
        baseline=baseline_path,
        baseline_note=note,
        source="results.jsonl",
    )
    report = build_report(ordered, corpus_keys=keys, gate=gate, meta=meta, k=k, judge=None)
    _write_report(out, report, gate)
    write_results(out, ordered)
    return report


def merge_from_aggregates(
    metrics_parts: Sequence[dict[str, Any]],
    *,
    planned_ids: Sequence[str],
    case_rows: dict[str, dict[str, Any]],
    out: Path,
    subset: str,
    baseline_path: Path,
    baseline_advisory: bool,
) -> dict[str, Any]:
    """Pool each split's Metrics from the parts' metrics.json (exact for rates/counts)."""
    pooled: dict[str, Metrics] = {}
    for split in SPLITS:
        items = [Metrics.model_validate(part["metrics"][split]) for part in metrics_parts]
        # Skip empty heldout from a part that had no heldout cases: model_validate still works
        # with cases=0 rates. Pooling n=0 rates is fine (rate(0,0) stays undefined).
        pooled[split] = pool_metrics(items)

    flat = pooled["all"].flat()
    gate = evaluate_gate(flat, baseline_path)
    note = baseline_note(gate, advisory=baseline_advisory)
    meta = _merged_meta(
        metrics_parts,
        cases=len(planned_ids),
        subset=subset,
        baseline=baseline_path,
        baseline_note=note,
        source="pooled_sufficient_statistics",
    )
    counts = Counter((case_rows[i]["category"], case_rows[i]["split"]) for i in planned_ids)
    report = {
        "meta": meta,
        "counts": {f"{c}/{s}": n for (c, s), n in sorted(counts.items())},
        "metrics": {name: m.model_dump(mode="json") for name, m in pooled.items()},
        "flat": {name: m.flat() for name, m in pooled.items()},
        "gate": gate.model_dump(mode="json"),
        "cases": [
            {
                "id": case_rows[i]["id"],
                "category": case_rows[i]["category"],
                "split": case_rows[i]["split"],
                "surface": case_rows[i].get("surface"),
                "failures": list(case_rows[i].get("failures") or []),
            }
            for i in planned_ids
        ],
        "_metrics_objects": pooled,
    }
    _write_report(out, report, gate)
    return report


def merge_reports(
    part_dirs: Sequence[Path],
    *,
    out: Path,
    require_commit: str,
    subset: str = "all",
    datasets: Path = DATASETS_DIR,
    baseline: Path | None = None,
    baseline_advisory: bool = False,
    k: int = 5,
) -> dict[str, Any]:
    metrics_parts = [_load_metrics(p) for p in part_dirs]
    cases = load_cases(datasets)
    planned_ids = [c.id for c in cases]
    case_rows = validate_parts(
        metrics_parts, require_commit=require_commit, planned_ids=planned_ids
    )
    model = str(metrics_parts[0]["meta"]["model"])
    baseline_path = baseline or default_baseline(model, subset)
    if all(has_results(p) for p in part_dirs):
        return merge_from_results(
            part_dirs,
            metrics_parts,
            planned_ids=planned_ids,
            cases_for_keys=cases,
            out=out,
            subset=subset,
            baseline_path=baseline_path,
            baseline_advisory=baseline_advisory,
            k=k,
        )
    if any(has_results(p) for p in part_dirs):
        raise MergeError(
            "some parts have results.jsonl and some do not; re-run the older parts "
            "or remove results.jsonl so every part merges the same way"
        )
    return merge_from_aggregates(
        metrics_parts,
        planned_ids=planned_ids,
        case_rows=case_rows,
        out=out,
        subset=subset,
        baseline_path=baseline_path,
        baseline_advisory=baseline_advisory,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.merge", description=__doc__)
    parser.add_argument("parts", nargs="+", type=Path, help="Report directories to merge.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the merged report.")
    parser.add_argument(
        "--require-commit",
        required=True,
        help="Every part must record this git short commit (e.g. 7803d20).",
    )
    parser.add_argument("--subset", default="all", help="Subset label for the merged report.")
    parser.add_argument("--datasets", type=Path, default=DATASETS_DIR)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--baseline-advisory", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        report = merge_reports(
            args.parts,
            out=args.out,
            require_commit=args.require_commit,
            subset=args.subset,
            datasets=args.datasets,
            baseline=args.baseline,
            baseline_advisory=args.baseline_advisory,
            k=args.k,
        )
    except (MergeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    meta = report["meta"]
    print(
        f"wrote {args.out / 'report.md'} and {args.out / METRICS_NAME} "
        f"({meta['cases']} cases, source={meta['merged']['source']}, "
        f"tokens={(meta.get('provider') or {}).get('tokens')})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
