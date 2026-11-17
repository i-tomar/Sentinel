"""Char n-gram TF-IDF + logistic regression injection classifier (offline artifact).

The model file is trained by `python -m sentinel.guards.train_classifier` with a fixed seed
and committed under `artifacts/`. Loading it needs no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

InjectionGuardMode = Literal["heuristic", "classifier", "both"]

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "injection_classifier.joblib"
META_PATH = ARTIFACT_DIR / "injection_classifier.meta.json"

DEFAULT_SEED = 17
DEFAULT_CLASSIFIER_THRESHOLD = 0.5


def build_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=500,
                ),
            ),
        ]
    )


@dataclass
class ClassifierModel:
    """Loaded sklearn pipeline plus the decision threshold chosen on DEV."""

    pipeline: Pipeline
    threshold: float

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        proba = self.pipeline.predict_proba([text])[0]
        # classes_ may be [0, 1]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        idx = classes.index(1) if 1 in classes else len(classes) - 1
        return float(proba[idx])

    def flagged(self, text: str) -> bool:
        return self.score(text) >= self.threshold


def save_model(pipeline: Pipeline, threshold: float, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": float(threshold)}, path)
    META_PATH.write_text(
        json.dumps({"threshold": float(threshold), "path": path.name}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_model(path: Path = MODEL_PATH) -> ClassifierModel:
    if not path.is_file():
        raise FileNotFoundError(f"classifier artifact missing: {path}")
    payload: Any = joblib.load(path)
    if isinstance(payload, dict) and "pipeline" in payload:
        return ClassifierModel(
            pipeline=payload["pipeline"], threshold=float(payload.get("threshold", 0.5))
        )
    # Older plain pipeline dumps
    return ClassifierModel(pipeline=payload, threshold=DEFAULT_CLASSIFIER_THRESHOLD)


_CACHED: ClassifierModel | None = None


def get_classifier(path: Path = MODEL_PATH) -> ClassifierModel:
    global _CACHED
    if _CACHED is None or path != MODEL_PATH:
        model = load_model(path)
        if path == MODEL_PATH:
            _CACHED = model
        return model
    return _CACHED


def reset_classifier_cache() -> None:
    global _CACHED
    _CACHED = None
