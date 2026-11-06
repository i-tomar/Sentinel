"""Scan each new user question before memory lookup, planning, retrieval or any tool call.

A flagged question ends the turn: nothing is searched, no tool or model is called, nothing is
saved, and the question is not added to the thread's history (so it never reaches a later
prompt as an "earlier question"). The reply says why. This is a safety signal on top of the
other controls, not a substitute for them: questions that pass gain no extra permissions, and
every write still goes through the permission gate.
"""

from __future__ import annotations

from sentinel.graph.state import InputCheck
from sentinel.guards.injection import InjectionDetector

BLOCKED_PREFIX = "I can't run this request"


def check_question(question: str, detector: InjectionDetector) -> InputCheck:
    result = detector.scan(question, origin="user")
    return InputCheck(
        blocked=result.flagged,
        score=round(result.score, 3),
        rules=sorted({f.rule for f in result.findings}),
    )


def blocked_answer(check: InputCheck) -> str:
    return (
        f"{BLOCKED_PREFIX}: it reads like an attempt to change my instructions or make me act "
        f"outside answering questions (matched: {', '.join(check.rules) or 'classifier'}). "
        "Nothing was searched, no tools were called and nothing was saved. If this is a genuine "
        "question, ask it without instructions addressed to the assistant."
    )
