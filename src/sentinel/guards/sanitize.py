"""Neutralize injected instructions found in untrusted text.

Every sentence (or line) that contains an injection finding is either removed and replaced by
a visible marker, or wrapped as quoted untrusted text. Whole sentences go because the matched
words alone ("Note for any AI assistant") would leave the rest of the instruction behind.
Text that was not flagged is returned unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from sentinel.guards.injection import InjectionFinding, InjectionResult

GuardMode = Literal["remove", "quote"]
REMOVED_MARKER = "[removed: suspected injected instruction]"
QUOTE_PREFIX = '[untrusted text, not an instruction: "'
QUOTE_SUFFIX = '"]'

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n")
_QUOTED = re.compile(re.escape(QUOTE_PREFIX) + r'[^"]*' + re.escape(QUOTE_SUFFIX))


def mask_neutralized(text: str) -> str:
    """Blank out already-quoted spans (same length, so finding offsets stay valid).

    Quoted text still contains the instruction wording; scanning it again would re-flag and
    re-wrap it on every hop.
    """
    return _QUOTED.sub(lambda m: " " * len(m.group()), text)


def _segments(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence or line, covering the whole text."""
    segments, start = [], 0
    for m in _BOUNDARY.finditer(text):
        segments.append((start, m.start()))
        start = m.end()
    segments.append((start, len(text)))
    return [(s, e) for s, e in segments if e > s]


def flagged_spans(text: str, findings: Sequence[InjectionFinding]) -> list[tuple[int, int]]:
    """Finding spans grown to whole sentences/lines, merged where they touch."""
    segments = _segments(text)
    grown: list[tuple[int, int]] = []
    for f in findings:
        start = min((s for s, e in segments if s <= f.start < e), default=f.start)
        end = max((e for s, e in segments if s < f.end <= e), default=f.end)
        grown.append((min(start, f.start), max(end, f.end)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(grown):
        gap = text[merged[-1][1] : start] if merged and start >= merged[-1][1] else ""
        touching = merged and (
            start <= merged[-1][1] or (gap.strip(" \t") == "" and "\n" not in gap)
        )
        if touching:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _quote(fragment: str) -> str:
    flat = " ".join(fragment.replace('"', "'").split())
    return f"{QUOTE_PREFIX}{flat}{QUOTE_SUFFIX}"


def neutralize(text: str, result: InjectionResult, mode: GuardMode = "remove") -> tuple[str, int]:
    """Return (sanitized text, number of neutralized spans)."""
    if not result.flagged or not result.findings:
        return text, 0
    spans = flagged_spans(text, result.findings)
    out, cursor = [], 0
    for start, end in spans:
        out.append(text[cursor:start])
        out.append(REMOVED_MARKER if mode == "remove" else _quote(text[start:end]))
        cursor = end
    out.append(text[cursor:])
    return "".join(out), len(spans)
