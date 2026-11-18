"""Train the injection classifier from the training file + DEV cases only.

Never uses held-out / *-h* / *-h2* eval cases for fitting or threshold selection.

  uv run python -m sentinel.guards.train_classifier
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sentinel.guards.classifier import (
    DEFAULT_SEED,
    MODEL_PATH,
    build_pipeline,
    save_model,
)

_HELD_ID = re.compile(r"-h(?:2-|\d)", re.I)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "evals").is_dir():
            return parent
    return Path.cwd()


TRAIN_PATH = _repo_root() / "evals" / "training" / "injection_train.jsonl"
DATASETS_DIR = _repo_root() / "evals" / "datasets"


def _load_training_rows(path: Path = TRAIN_PATH) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append((str(obj["text"]), int(obj["label"])))
    return rows


def _dev_lines(path: Path) -> list[dict[str, object]]:
    """Parse only JSONL lines that declare split=dev and whose id is not held-out."""
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"split": "dev"' not in line and '"split":"dev"' not in line:
            continue
        obj = json.loads(line)
        case_id = str(obj.get("id", ""))
        if obj.get("split") != "dev" or _HELD_ID.search(case_id) or "-h2-" in case_id:
            continue
        out.append(obj)
    return out


def _extra_record_texts(extra_records: object) -> list[str]:
    texts: list[str] = []
    if not isinstance(extra_records, list):
        return texts
    for extra in extra_records:
        if not isinstance(extra, dict):
            continue
        record = extra.get("record")
        if not isinstance(record, dict):
            continue
        parts: list[str] = []
        for key in ("title", "summary", "description", "subject", "body"):
            val = record.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        for msg in record.get("messages") or []:
            if isinstance(msg, dict) and isinstance(msg.get("body"), str):
                parts.append(msg["body"])
        for comment in record.get("comments") or []:
            if isinstance(comment, dict) and isinstance(comment.get("body"), str):
                parts.append(comment["body"])
        if parts:
            texts.append("\n".join(parts))
    return texts


def load_dev_examples() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(fit rows from DEV, threshold-tuning rows from DEV). Labels: 1=attack, 0=benign."""
    fit: list[tuple[str, int]] = []
    tune: list[tuple[str, int]] = []
    for path in sorted(DATASETS_DIR.glob("*.jsonl")):
        for obj in _dev_lines(path):
            category = str(obj.get("category", ""))
            question = str(obj.get("question", "")).strip()
            extras = _extra_record_texts(obj.get("extra_records"))
            if category == "adversarial_user":
                if question:
                    fit.append((question, 1))
                    tune.append((question, 1))
            elif category == "adversarial_retrieved":
                for text in extras or ([question] if question else []):
                    fit.append((text, 1))
                    tune.append((text, 1))
            elif category == "benign_lookalike":
                surface = obj.get("surface")
                if surface == "input" and question:
                    fit.append((question, 0))
                    tune.append((question, 0))
                elif surface == "retrieved":
                    for text in extras or ([question] if question else []):
                        fit.append((text, 0))
                        tune.append((text, 0))
            elif category in ("normal", "edge", "memory_freshness", "tool_failure") and question:
                # extra benign signal for fitting only
                fit.append((question, 0))
    return fit, tune


def _score(pipeline: object, text: str) -> float:
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

    if not isinstance(pipeline, Pipeline):
        raise TypeError("expected an sklearn Pipeline")
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.named_steps["clf"].classes_)
    idx = classes.index(1) if 1 in classes else len(classes) - 1
    return float(proba[idx])


def pick_threshold(
    pipeline: object,
    tune: list[tuple[str, int]],
    *,
    max_fp: float = 0.05,
) -> float:
    """Threshold with DEV benign FP <= max_fp; break ties by higher attack recall."""
    benign = [t for t, y in tune if y == 0]
    attacks = [t for t, y in tune if y == 1]
    if not benign:
        return 0.5
    scores_b = [_score(pipeline, t) for t in benign]
    scores_a = [_score(pipeline, t) for t in attacks]
    best_t, best_key = 0.5, (-1.0, -1.0)
    for raw in range(5, 96):
        thr = raw / 100.0
        fp = sum(s >= thr for s in scores_b) / len(scores_b)
        if fp > max_fp:
            continue
        recall = (sum(s >= thr for s in scores_a) / len(scores_a)) if scores_a else 0.0
        key = (recall, -fp)
        if key > best_key:
            best_key = key
            best_t = thr
    return best_t


def main() -> None:
    rng = random.Random(DEFAULT_SEED)  # noqa: S311 — fixed seed for reproducible training
    train_rows = _load_training_rows()
    fit_dev, tune_dev = load_dev_examples()
    rows = train_rows + fit_dev
    rng.shuffle(rows)
    texts = [t for t, _ in rows]
    labels = [y for _, y in rows]
    pipeline = build_pipeline(DEFAULT_SEED)
    pipeline.fit(texts, labels)
    threshold = pick_threshold(pipeline, tune_dev, max_fp=0.05)
    save_model(pipeline, threshold, MODEL_PATH)
    benign = [t for t, y in tune_dev if y == 0]
    attacks = [t for t, y in tune_dev if y == 1]
    fp = sum(_score(pipeline, t) >= threshold for t in benign) / max(1, len(benign))
    rec = sum(_score(pipeline, t) >= threshold for t in attacks) / max(1, len(attacks))
    print(
        f"wrote {MODEL_PATH} threshold={threshold:.2f} "
        f"fit_n={len(rows)} tune_benign={len(benign)} "
        f"dev_fp={fp:.3f} dev_attack_recall={rec:.3f}"
    )


if __name__ == "__main__":
    main()
