"""Last check before an answer is shown: redact PII from the answer and its citations.

Evidence is already redacted when records enter the agent's state, so this mostly catches PII
the model produced itself (or echoed from the question). Citation markers, ids and URLs are
untouched, so every [n] still resolves.
"""

from __future__ import annotations

from collections.abc import Sequence

from sentinel.citations import Citation
from sentinel.guards.redaction import redact_citation, redaction_kinds


def guard_output(
    answer: str, citations: Sequence[Citation]
) -> tuple[str, list[Citation], list[str]]:
    """(redacted answer, redacted citations, PII kinds removed from the answer)."""
    clean, kinds = redaction_kinds(answer)
    return clean, [redact_citation(c) for c in citations], kinds
