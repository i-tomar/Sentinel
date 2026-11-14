"""One-line, redacted descriptions of why an eval case could not run.

Reports need to say *why* a case failed (rate limit, daily token limit, server error) without
ever exposing a credential. `safe_error_message` keeps the exception type and message (plus
one level of cause), then removes: the configured secret values, anything shaped like an API
key or bearer token, provider organisation ids, and the PII the redactor detects. The result
is a single line of at most `MAX_CHARS` characters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from sentinel.guards.redaction import redact_str

MAX_CHARS = 300
MASK = "<SECRET>"
_SECRET_SHAPES = (
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),  # Groq keys
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),  # OpenAI-style keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|x-api-key|access[_-]?token|secret)"
        r"([\"']?\s*[:=]\s*[\"']?)[^\s\"',;}]+"
    ),
)
_ORG_ID = re.compile(r"\borg_[A-Za-z0-9]{6,}")


def leaf_exception(exc: BaseException) -> BaseException:
    """The first underlying error inside (nested) exception groups, which the async task
    groups around the agent wrap failures in; any other exception is returned as is."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".rstrip(": ")
    cause = exc.__cause__ or exc.__context__
    if cause is not None and str(cause) and str(cause) not in text:
        text += f" (caused by {type(cause).__name__}: {cause})"
    return text


def scrub(text: str, secrets: Iterable[str] = ()) -> str:
    for value in secrets:
        if value:
            text = text.replace(value, MASK)
    for shape in _SECRET_SHAPES:
        text = shape.sub(
            lambda m: f"{m.group(1)}{m.group(2)}{MASK}" if m.re.groups == 2 else MASK, text
        )
    text = _ORG_ID.sub("<ORG>", text)
    text = redact_str(text)
    text = " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 3] + "..."


def safe_error_message(exc: BaseException, secrets: Iterable[str] = ()) -> str:
    return scrub(_describe(leaf_exception(exc)), secrets)
