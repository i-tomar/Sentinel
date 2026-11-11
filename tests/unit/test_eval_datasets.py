"""The eval datasets are well-formed, versioned, and keep an honest held-out split."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evals.datasets import DATASETS_DIR, DatasetError, file_hashes, load_cases, read_manifest
from evals.harness import corpus_keys
from evals.schema import EvalCase
from servers.fixtures import failure_modes, generate
from servers.fixtures.generate import build_dataset
from tests.guard_samples import ADVERSARIAL_USER, BENIGN_USER

pytestmark = pytest.mark.unit

CASES = load_cases()
DATASET = build_dataset()
PAYLOADS = (
    generate.HTML_COMMENT_PAYLOAD,
    generate.FOOTER_PAYLOAD,
    generate.LINK_TEXT_PAYLOAD,
    generate.DELIMITER_PAYLOAD,
    generate.TICKET_COMMENT_PAYLOAD,
    generate.SYSTEM_PREFIX_TICKET_PAYLOAD,
    generate.BASE64_PAYLOAD,
    generate.EMAIL_BODY_PAYLOAD,
    generate.TOOL_JSON_PAYLOAD,
    generate.HINGLISH_PAYLOAD,
    generate.HINDI_PAYLOAD,
    generate.SYSTEM_PREFIX_EMAIL_PAYLOAD,
)
MARKERS = {p.payload_marker for p in DATASET.manifest.poisoned} | {
    s for p in DATASET.manifest.poisoned for s in p.forbidden_strings
}


def test_manifest_matches_the_files() -> None:
    manifest = read_manifest()
    assert manifest.files == file_hashes()
    assert manifest.version


def test_an_unrecorded_edit_is_refused(tmp_path: Path) -> None:
    for name in ("manifest.json", "edge.jsonl"):
        (tmp_path / name).write_bytes((DATASETS_DIR / name).read_bytes())
    data = (tmp_path / "edge.jsonl").read_text(encoding="utf-8")
    (tmp_path / "edge.jsonl").write_text(data.replace("Pune", "Delhi"), encoding="utf-8")
    with pytest.raises(DatasetError, match="manifest"):
        load_cases(tmp_path)


def test_every_category_and_required_kind_is_covered() -> None:
    by_category = Counter(c.category for c in CASES)
    for category in (
        "normal",
        "edge",
        "adversarial_user",
        "adversarial_retrieved",
        "tool_failure",
        "memory_freshness",
        "benign_lookalike",
    ):
        assert by_category[category] >= 4, category
    tags = Counter(t for c in CASES for t in c.tags)
    for tag in (
        "single-hop",
        "multi-hop",
        "stale-vs-current",
        "follow-up",
        "no-answer",
        "paraphrase",
        "hindi",
        "hinglish",
        "encoded",
        "multi-turn",
        "security-discussion",
        "write-request",
        "pre-approved-claim",
        "outside-domain",
        "pii",
        "negation",
    ):
        assert tags[tag] >= 1, tag
    assert {c.fixture_overrides for c in CASES if c.category == "tool_failure"} >= {
        "gmail=timeout",
        "jira=rate_limit",
        "notion.get_page=malformed",
        "*=empty",
        "*=error_500",
        "jira.search_issues=invalid_schema",
    }


def test_attack_and_benign_cases_have_a_held_out_split() -> None:
    for category, surface in (
        ("adversarial_user", "input"),
        ("adversarial_retrieved", "retrieved"),
        ("benign_lookalike", "input"),
        ("benign_lookalike", "retrieved"),
    ):
        splits = Counter(c.split for c in CASES if c.category == category and c.surface == surface)
        assert splits["heldout"] >= 3, (category, surface)
        assert splits["dev"] >= 3, (category, surface)


def _texts(case: EvalCase) -> list[str]:
    bodies = [
        str(m.get("body", ""))
        for extra in case.extra_records
        for m in [extra.record, *extra.record.get("messages", [])]
    ]
    return [case.question, *case.thread_setup, *bodies]


@pytest.mark.parametrize("case", [c for c in CASES if c.split == "heldout"], ids=lambda c: c.id)
def test_held_out_cases_do_not_copy_fixture_payloads_or_detector_samples(case: EvalCase) -> None:
    samples = set(ADVERSARIAL_USER) | set(BENIGN_USER)
    for text in _texts(case):
        assert text not in samples
        assert not any(payload in text for payload in PAYLOADS)
        assert not any(marker in text for marker in MARKERS)


def test_expected_records_exist() -> None:
    known = corpus_keys(CASES, DATASET)
    for case in CASES:
        e = case.expected
        for key in (*e.expected_records, *e.required_citations, *e.forbidden_citations):
            assert key in known, (case.id, key)


def test_failure_rules_parse_and_judge_cases_carry_a_rubric() -> None:
    for case in CASES:
        failure_modes.parse_spec(case.fixture_overrides)
        if case.needs_judge:
            assert case.expected.answer_rubric
    assert sum(c.needs_judge for c in CASES) >= 4
