"""PII redaction. Regex recognizers only for now (used for traces); Phase 3 adds presidio.

Findings carry kind and position, never the matched value, so they are safe to log.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

PiiKind = Literal["EMAIL", "PHONE_IN", "PAN", "AADHAAR"]

# Order matters: Aadhaar (12 digits) before phone so a 12-digit run is not half-read as a phone.
_RECOGNIZERS: tuple[tuple[PiiKind, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("PHONE_IN", re.compile(r"(?<![\w+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
)


class PiiFinding(BaseModel):
    kind: PiiKind
    start: int
    end: int


def redact(text: str) -> tuple[str, list[PiiFinding]]:
    """Replace PII with `<KIND>` placeholders. Positions refer to the original text."""
    findings: list[PiiFinding] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _RECOGNIZERS:
        for m in pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            taken.append((m.start(), m.end()))
            findings.append(PiiFinding(kind=kind, start=m.start(), end=m.end()))
    findings.sort(key=lambda f: f.start)
    out, cursor = [], 0
    for f in findings:
        out.append(text[cursor : f.start])
        out.append(f"<{f.kind}>")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out), findings


def redact_text(text: str) -> str:
    return redact(text)[0]
