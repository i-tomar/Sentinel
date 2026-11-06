"""PII redaction: Presidio recognizers plus custom ones for Indian identifiers.

Offline by design: Presidio runs on a blank spaCy pipeline (no model download, so no
name/location NER), and its email and URL recognizers are not loaded because they fetch the
public-suffix list over the network on first use; a local email pattern replaces the first,
and URLs are deliberately kept (citations need them). Only an allowlist of Presidio's
built-in recognizers is enabled, because others (dates, UUIDs, US licence numbers) fire on
ticket ids, page ids and dates.

Findings carry kind and position, never the matched value, so they are safe to log.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, cast

import spacy
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    CryptoRecognizer,
    IbanRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    UsSsnRecognizer,
)
from pydantic import BaseModel

PiiKind = Literal[
    "EMAIL",
    "PHONE_IN",
    "PHONE",
    "PAN",
    "AADHAAR",
    "CREDIT_CARD",
    "IBAN",
    "IP_ADDRESS",
    "US_SSN",
    "CRYPTO",
]

MIN_SCORE = 0.35
_ENTITY_TO_KIND: dict[str, PiiKind] = {
    "EMAIL": "EMAIL",
    "PHONE_IN": "PHONE_IN",
    "PHONE_NUMBER": "PHONE",
    "PAN": "PAN",
    "AADHAAR": "AADHAAR",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "US_SSN",
    "CRYPTO": "CRYPTO",
}


class PiiFinding(BaseModel):
    kind: PiiKind
    start: int
    end: int


class _BlankSpacyEngine(SpacyNlpEngine):
    """Presidio NLP engine backed by `spacy.blank("en")`: tokenization only, no download."""

    def __init__(self) -> None:
        super().__init__(models=[{"lang_code": "en", "model_name": "blank"}])
        # presidio declares `nlp` as None until load(); we provide the pipeline directly
        self.nlp = cast(Any, {"en": spacy.blank("en")})

    def load(self) -> None:
        pass


def _custom(entity: str, regex: str, score: float) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=entity,
        name=f"Sentinel{entity.title().replace('_', '')}Recognizer",
        patterns=[Pattern(name=entity.lower(), regex=regex, score=score)],
    )


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    engine = _BlankSpacyEngine()
    registry = RecognizerRegistry(supported_languages=["en"])
    for recognizer in (
        CreditCardRecognizer(),
        CryptoRecognizer(),
        IbanRecognizer(),
        IpRecognizer(),
        PhoneRecognizer(),
        UsSsnRecognizer(),
        _custom("EMAIL", r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", 0.9),
        # Digit-group boundaries: never match inside a longer number such as a card number.
        _custom(
            "PHONE_IN",
            r"(?<![\w+])(?<!\d[ -])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?![ -]?\d)",
            0.85,
        ),
        _custom("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.8),
        _custom("AADHAAR", r"(?<![\d-])(?<!\d )[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)", 0.8),
    ):
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])


def _resolve(results: list[Any]) -> list[PiiFinding]:
    """Keep non-overlapping detections, preferring higher score, then longer span."""
    ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start), r.start))
    kept: list[PiiFinding] = []
    for r in ranked:
        kind = _ENTITY_TO_KIND.get(r.entity_type)
        if kind is None or r.score < MIN_SCORE:
            continue
        if any(r.start < k.end and k.start < r.end for k in kept):
            continue
        kept.append(PiiFinding(kind=kind, start=r.start, end=r.end))
    return sorted(kept, key=lambda f: f.start)


def find_pii(text: str) -> list[PiiFinding]:
    if not text:
        return []
    return _resolve(_analyzer().analyze(text=text, language="en"))


def redact(text: str) -> tuple[str, list[PiiFinding]]:
    """Replace PII with `<KIND>` placeholders. Positions refer to the original text."""
    findings = find_pii(text)
    out, cursor = [], 0
    for f in findings:
        out.append(text[cursor : f.start])
        out.append(f"<{f.kind}>")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out), findings


def redact_text(text: str) -> str:
    return redact(text)[0]
