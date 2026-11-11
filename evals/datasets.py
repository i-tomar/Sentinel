"""Load and version the eval datasets (evals/datasets/*.jsonl).

`manifest.json` records a dataset version and the sha256 of every file. Loading checks the
hashes, so a dataset cannot change without the version (and the manifest) changing with it;
reports record the version and the hashes they ran on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from evals.schema import EvalCase

DATASETS_DIR = Path(__file__).parent / "datasets"
MANIFEST = "manifest.json"


class DatasetError(Exception):
    pass


class Manifest(BaseModel):
    version: str
    files: dict[str, str]
    """File name -> sha256 of its bytes."""


def file_hashes(directory: Path = DATASETS_DIR) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.glob("*.jsonl"))
    }


def write_manifest(version: str, directory: Path = DATASETS_DIR) -> Manifest:
    """Owner step after editing a dataset: bump the version and record the new hashes."""
    manifest = Manifest(version=version, files=file_hashes(directory))
    (directory / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(directory: Path = DATASETS_DIR) -> Manifest:
    path = directory / MANIFEST
    if not path.exists():
        raise DatasetError(f"no {MANIFEST} in {directory}")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_cases(directory: Path = DATASETS_DIR, *, verify: bool = True) -> list[EvalCase]:
    if verify:
        manifest = read_manifest(directory)
        actual = file_hashes(directory)
        if actual != manifest.files:
            changed = sorted(
                set(actual) ^ set(manifest.files)
                | {k for k in actual if manifest.files.get(k) != actual[k]}
            )
            raise DatasetError(
                f"dataset files changed without a manifest update: {', '.join(changed)} "
                "(bump the version with evals.datasets.write_manifest)"
            )
    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise DatasetError(f"{path.name}:{number}: {exc}") from None
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DatasetError(f"duplicate case ids: {', '.join(duplicates)}")
    return cases
