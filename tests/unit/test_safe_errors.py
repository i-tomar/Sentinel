"""Per-case error messages say why a case failed without exposing secrets or detected PII."""

from __future__ import annotations

import pytest

from evals.safe_errors import MAX_CHARS, leaf_exception, safe_error_message, scrub

pytestmark = pytest.mark.unit

KEY = "gsk_" + "Zq9" * 12


def test_a_rate_limit_message_keeps_the_useful_part() -> None:
    exc = RuntimeError(
        "Error code: 429 - Rate limit reached for model openai/gpt-oss-120b in organization "
        "org_01abcdefghij on tokens per day (TPD): Limit 200000, Used 199500, Requested 1500. "
        "Please try again in 7m12s."
    )
    text = safe_error_message(exc)
    assert text.startswith("RuntimeError: Error code: 429 - Rate limit reached")
    assert "tokens per day (TPD)" in text
    assert "try again in 7m12s" in text
    assert "org_01abcdefghij" not in text
    assert "<ORG>" in text


def test_configured_secrets_are_removed() -> None:
    exc = ValueError(f"request with key {KEY} failed")
    text = safe_error_message(exc, secrets=(KEY,))
    assert KEY not in text
    assert "<SECRET>" in text


@pytest.mark.parametrize(
    "leak",
    [
        KEY,  # a Groq key that was not configured here
        "sk-proj-abcdefghijklmnop1234",
        "Bearer abcdefghijklmnopqrstuvwxyz012345",
        "api_key=abc123def456ghi789",
        '"authorization": "xyzxyzxyzxyzxyz"',
    ],
)
def test_key_shaped_text_is_removed_even_if_not_configured(leak: str) -> None:
    text = scrub(f"boom: {leak} end")
    secret_part = leak.split()[-1].split("=")[-1].strip('"')
    assert secret_part not in text
    assert "<SECRET>" in text


def test_detected_pii_is_redacted() -> None:
    text = scrub("failed for priya.sharma@acme.test, call +91 98200 11223")
    assert "priya.sharma@acme.test" not in text
    assert "98200 11223" not in text
    assert "<EMAIL>" in text


def test_the_cause_is_included_once() -> None:
    try:
        try:
            raise ConnectionResetError("connection reset by peer")
        except ConnectionResetError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as exc:
        text = safe_error_message(exc)
    assert text == (
        "RuntimeError: Connection error. (caused by ConnectionResetError: connection reset by peer)"
    )


def test_the_message_is_one_short_line() -> None:
    text = safe_error_message(RuntimeError("line one\nline two\t" + "x" * 1000))
    assert "\n" not in text
    assert len(text) <= MAX_CHARS
    assert text.endswith("...")
    assert text.startswith("RuntimeError: line one line two")


def test_exception_groups_are_unwrapped_to_the_real_error() -> None:
    inner = ValueError("Error code: 400 - bad request")
    group = ExceptionGroup("unhandled errors in a TaskGroup", [ExceptionGroup("x", [inner])])
    assert leaf_exception(group) is inner
    assert safe_error_message(group) == "ValueError: Error code: 400 - bad request"
    plain = RuntimeError("plain")
    assert leaf_exception(plain) is plain


def test_an_empty_message_keeps_the_type() -> None:
    assert safe_error_message(TimeoutError()) == "TimeoutError"
