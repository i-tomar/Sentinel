"""How `python -m evals.run` records runs: a timestamped log per run, a lock on the output
directory, and a partial report when the suite stops early. Mock LLM only, offline."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evals import run as eval_run
from evals.harness import TokenBudgetError
from evals.runlog import RunInProgressError, RunLog, exclusive_run, log_path, run_id
from evals.schema import CaseResult

pytestmark = pytest.mark.integration


def _logs(out: Path) -> list[Path]:
    return sorted((out / "logs").glob("eval-*.log"))


def test_run_ids_name_the_time_model_subset_and_process() -> None:
    rid = run_id(datetime(2026, 11, 14, 12, 6, 41, tzinfo=UTC), "groq:openai/gpt-oss-120b", "fast")
    assert re.fullmatch(r"20261114T120641Z-groq-openai-gpt-oss-120b-fast-p\d+", rid)
    assert log_path(Path("reports"), rid) == Path("reports/logs") / f"eval-{rid}.log"


def test_the_run_log_writes_and_flushes_each_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "logs" / "eval-x.log"
    with RunLog(path) as log:
        log.line("[  1/2] norm-001   ok")
        assert path.read_text(encoding="utf-8") == "[  1/2] norm-001   ok\n"  # already on disk
        log.line("stopped: budget", err=True)
    captured = capsys.readouterr()
    assert captured.out == "[  1/2] norm-001   ok\n"
    assert captured.err == "stopped: budget\n"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "[  1/2] norm-001   ok",
        "stopped: budget",
    ]


def test_only_one_run_can_hold_an_output_directory(tmp_path: Path) -> None:
    with (
        exclusive_run(tmp_path),
        pytest.raises(RunInProgressError, match="another eval run"),
        exclusive_run(tmp_path),  # the second run: refused while the first holds the lock
    ):
        pass
    with exclusive_run(tmp_path):  # released when the first run ended
        pass


def test_each_run_gets_its_own_log_and_earlier_logs_are_kept(tmp_path: Path) -> None:
    args = ["--case", "norm-001", "--out", str(tmp_path), "--report-only"]
    assert eval_run.main(args) == 0
    (first,) = _logs(tmp_path)
    first_text = first.read_text(encoding="utf-8")
    assert "[  1/1] norm-001   ok" in first_text
    assert "gate: " in first_text
    rid = first.stem.removeprefix("eval-")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert f"run id {rid}" in report
    meta = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["meta"]
    assert meta["run_id"] == rid
    assert meta["partial"] is None

    assert eval_run.main(args) == 0  # a second run, as when a command is accidentally repeated
    logs = _logs(tmp_path)
    assert len(logs) == 2
    assert first.read_text(encoding="utf-8") == first_text  # not overwritten


def test_a_run_refuses_to_start_while_another_holds_the_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with exclusive_run(tmp_path):
        code = eval_run.main(["--case", "norm-001", "--out", str(tmp_path)])
    assert code == 1
    assert "another eval run is writing to" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_a_stopped_run_still_writes_a_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = CaseResult.model_validate(
        {"case_id": "norm-001", "category": "normal", "answer": "x", "split": "dev"}
    )

    async def stops_after_one(*_args: Any, **_kwargs: Any) -> list[CaseResult]:
        raise TokenBudgetError("token budget: 900 of 1000 tokens used", [finished], 3)

    monkeypatch.setattr(eval_run, "run_suite", stops_after_one)
    code = eval_run.main(["--case", "norm-001", "--out", str(tmp_path), "--baseline-advisory"])
    assert code == 1  # a partial run never passes
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "PARTIAL RUN: stopped after 1 of 3 cases (token budget: 900 of 1000" in report
    assert "do not compare them with a baseline" in report
    meta = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["meta"]
    assert meta["partial"] == {
        "reason": "token budget: 900 of 1000 tokens used",
        "completed": 1,
        "planned": 3,
    }
    (log,) = _logs(tmp_path)
    text = log.read_text(encoding="utf-8")
    assert "stopped: stopped after 1/3 cases: token budget" in text
    assert "partial run: 1 of 3 cases" in text


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(SystemExit, match="--token-budget"):
        eval_run.main(["--case", "norm-001", "--token-budget", "0"])
