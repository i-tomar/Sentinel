"""PII redaction (presidio + custom recognizers). Written before the presidio implementation."""

from __future__ import annotations

import pytest

from sentinel.guards.pii import PiiFinding, redact, redact_text
from servers.fixtures.generate import PII_SEEDS, build_dataset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", PII_SEEDS, ids=lambda s: f"{s.kind}-{s.record_id}")
def test_every_seeded_pii_value_is_redacted(seed: object) -> None:
    value = seed.value  # type: ignore[attr-defined]
    redacted, findings = redact(f"Contact details: {value}, thanks.")
    assert value not in redacted
    assert findings
    assert all(value not in f.model_dump_json() for f in findings)  # findings carry no values


def test_seeded_records_are_clean_after_redaction() -> None:
    ds = build_dataset()
    texts = [r.model_dump_json() for r in (*ds.notion, *ds.jira, *ds.gmail)]
    redacted = "\n".join(redact_text(t) for t in texts)
    for seed in PII_SEEDS:
        assert seed.value not in redacted, seed.kind


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail priya.sharma@acme.test now", "EMAIL"),
        ("call +91 98450 10001 today", "PHONE_IN"),
        ("call 09845010001 today", "PHONE_IN"),
        ("PAN ABCPS1234K on file", "PAN"),
        ("Aadhaar 2345 6789 0123 attached", "AADHAAR"),
        ("Aadhaar 234567890123 attached", "AADHAAR"),
        ("card 4111 1111 1111 1111 expires", "CREDIT_CARD"),
        ("IBAN GB82 WEST 1234 5698 7654 32 for payment", "IBAN"),
        ("US office +1 212 555 0100 ext 4", "PHONE"),
        ("server 10.20.30.40 leaked", "IP_ADDRESS"),
        ("SSN 078-05-1120 on the form", "US_SSN"),
    ],
)
def test_detects_each_kind(text: str, kind: str) -> None:
    redacted, findings = redact(text)
    assert [f.kind for f in findings] == [kind], findings
    assert f"<{kind}>" in redacted


@pytest.mark.parametrize(
    "text",
    [
        "PROJ-142 is blocked by PROJ-120; see auth-migration-v2.",
        "Invoice INV-2026-1043 for INR 4,85,000 is due 2026-11-15.",
        "Updated 2026-10-27T09:00:00+00:00, cutover 2026-11-20, window 02:00-04:00 IST.",
        "Pro plan is $59 per seat; ARR forecast USD 4.2M; click-through 41% to 50%.",
        "Notion page 0b8c7e9a-1d2f-4a3b-9c8d-7e6f5a4b3c2d and thread 18c2f0a9b7d3e4f1.",
        "Qdrant v1.20.0, Langfuse 4.76.0, port 6333, 120 seats, renewal 2027-01-31.",
        "Card ending 4111 1111 1111 1112 is invalid (fails the Luhn check).",
        "See https://jira.fixture.acme.test/browse/PROJ-142 for details.",
    ],
)
def test_ids_dates_amounts_and_urls_are_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_positions_refer_to_the_original_text() -> None:
    text = "a@acme.test and b@acme.test"
    redacted, findings = redact(text)
    assert redacted == "<EMAIL> and <EMAIL>"
    assert [(f.start, f.end) for f in findings] == [(0, 11), (16, 27)]
    assert all(isinstance(f, PiiFinding) for f in findings)


def test_redaction_is_idempotent() -> None:
    once = redact_text("Sofia +91 98200 11223, sofia.alvarez@acme.test, PAN ABCPS1234K")
    assert redact_text(once) == once


def test_empty_text() -> None:
    assert redact("") == ("", [])
