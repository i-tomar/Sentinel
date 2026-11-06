from __future__ import annotations

import pytest

from sentinel.guards.pii import redact, redact_text
from servers.fixtures.generate import PII_SEEDS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seed", PII_SEEDS, ids=lambda s: s.kind)
def test_every_seeded_pii_value_is_redacted(seed: object) -> None:
    value = seed.value  # type: ignore[attr-defined]
    text = f"Contact details: {value}, thanks."
    redacted, findings = redact(text)
    assert value not in redacted
    assert findings
    assert all(value not in f.model_dump_json() for f in findings)  # findings hold no values


def test_kinds_and_placeholders() -> None:
    text = (
        "Mail priya.sharma@acme.test or call +91 98450 10001. "
        "PAN ABCPS1234K, Aadhaar 2345 6789 0123."
    )
    redacted, findings = redact(text)
    assert redacted == "Mail <EMAIL> or call <PHONE_IN>. PAN <PAN>, Aadhaar <AADHAAR>."
    assert [f.kind for f in findings] == ["EMAIL", "PHONE_IN", "PAN", "AADHAAR"]


def test_non_pii_is_untouched() -> None:
    text = "PROJ-142 is blocked; invoice INV-2026-1043 for INR 4,85,000 is due 2026-11-15."
    assert redact_text(text) == text
