"""Verify a draft answer against the evidence that was actually retrieved.

- Markers that match no evidence are removed and reported (no invented citations survive).
- Factual sentences without a marker are reported.
- A cited sentence whose content words mostly do not appear in the cited evidence is reported
  as unsupported (lexical check; an LLM judge can replace it later).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sentinel.citations import Citation, cited_numbers
from sentinel.graph.nodes.answerer import is_insufficient
from sentinel.graph.state import CitationReport, Evidence, UnsupportedClaim
from sentinel.llm import INSUFFICIENT_PREFIX, PARTIAL_PREFIX
from sentinel.memory.freshness import support as memory_support
from sentinel.text import content_tokens, split_sentences

SUPPORT_THRESHOLD = 0.5
_MARKER = re.compile(r"\[(\d{1,3})\]")
_LEADING_MARKERS = re.compile(r"^((?:\s*\[\d{1,3}\])+)\s*")


def _sentences_with_markers(text: str) -> list[str]:
    """Split into sentences, re-attaching markers written after the period ("... done. [2]")."""
    out: list[str] = []
    for sentence in split_sentences(text):
        lead = _LEADING_MARKERS.match(sentence)
        if lead and out:
            out[-1] = f"{out[-1]} {lead.group(1).strip()}"
            sentence = sentence[lead.end() :]
        if sentence:
            out.append(sentence)
    return out


def check_answer(
    draft: str, evidence: Sequence[Evidence], unverified_memory: Sequence[str] = ()
) -> tuple[str, list[Citation], CitationReport]:
    """Return (final answer, citations actually used, report).

    `unverified_memory` holds values of unverified memory notes shown to the model; a
    sentence that restates one without support from cited evidence is removed.
    """
    by_n = {ev.citation.n: ev for ev in evidence}
    unknown = [n for n in cited_numbers(draft) if n not in by_n]
    answer = _MARKER.sub(lambda m: "" if int(m.group(1)) in unknown else m.group(0), draft)
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer).strip()

    report = CitationReport(unknown=unknown)
    if not is_insufficient(answer):
        for sentence in _sentences_with_markers(answer):
            cited = [n for n in cited_numbers(sentence) if n in by_n]
            words = content_tokens(_MARKER.sub("", sentence))
            if not words or sentence.startswith((PARTIAL_PREFIX, INSUFFICIENT_PREFIX)):
                continue
            support = 0.0
            if cited:
                support_words = set().union(
                    *(
                        content_tokens(f"{c.doc_id}\n{c.title}\n{by_n[n].text}")
                        for n in cited
                        for c in (by_n[n].citation,)
                    )
                )
                support = len(words & support_words) / len(words)
            if support < SUPPORT_THRESHOLD and any(
                memory_support(value, sentence) >= SUPPORT_THRESHOLD for value in unverified_memory
            ):
                report.unverified_memory_claims.append(sentence)
                continue
            if not cited:
                report.uncited_sentences.append(sentence)
            elif support < SUPPORT_THRESHOLD:
                report.unsupported.append(
                    UnsupportedClaim(sentence=sentence, cited=cited, support=round(support, 3))
                )
    for sentence in report.unverified_memory_claims:
        answer = answer.replace(sentence, "")
    if report.unverified_memory_claims:
        answer = re.sub(r"[ \t]{2,}", " ", answer).strip() or INSUFFICIENT_PREFIX + "."
        answer += (
            f"\n\n(Removed {len(report.unverified_memory_claims)} statement(s) that only "
            "unverified memory supports.)"
        )
    if unknown:
        listed = ", ".join(str(n) for n in unknown)
        answer += f"\n\n(Removed citation number(s) {listed}: no retrieved source has them.)"
    used = [by_n[n].citation for n in cited_numbers(answer) if n in by_n]
    return answer, used, report
