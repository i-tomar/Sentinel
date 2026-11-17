"""Persist and load per-case `CaseResult`s for mergeable eval runs.

`reports/metrics.json` and `report.md` keep only case ids and failure reasons (they must not
carry answers or record text). The companion `results.jsonl` holds one `CaseResult` per line
so a later merge can recompute every metric from the cases themselves. The file lives under
`reports/` (gitignored) and may contain answers and record text; do not publish it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from evals.schema import CaseResult

RESULTS_NAME = "results.jsonl"


def results_path(out: Path) -> Path:
    return out / RESULTS_NAME


def write_results(out: Path, results: Sequence[CaseResult]) -> Path:
    """Write one CaseResult JSON object per line (UTF-8, LF)."""
    path = results_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() for r in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def load_results(path: Path) -> list[CaseResult]:
    """Load CaseResults from a results.jsonl file (or a directory that contains one)."""
    file = path / RESULTS_NAME if path.is_dir() else path
    if not file.exists():
        raise FileNotFoundError(f"no {RESULTS_NAME} at {file}")
    results: list[CaseResult] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            results.append(CaseResult.model_validate_json(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{file.name}:{number}: {exc}") from None
    return results


def has_results(out: Path) -> bool:
    return results_path(out).is_file()


def ordered_by_ids(results: Iterable[CaseResult], ids: Sequence[str]) -> list[CaseResult]:
    """Return results in `ids` order; raises if any id is missing or extra."""
    by_id = {r.case_id: r for r in results}
    if len(by_id) != len(list(results)):
        raise ValueError("duplicate case ids in results")
    missing = [i for i in ids if i not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(f"case id mismatch: missing={missing!r} extra={extra!r}")
    return [by_id[i] for i in ids]
