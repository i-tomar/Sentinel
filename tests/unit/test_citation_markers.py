"""Citation styles other models use are read as `[n]` (gpt-oss cites in lenticular brackets).

Non-ASCII brackets are written as escapes: U+3010/U+3011 lenticular, U+FF3B/U+FF3D full-width
square, U+2020 dagger, U+FF11 full-width digit one.
"""

from __future__ import annotations

import pytest

from sentinel.citations import cited_numbers, normalize_markers
from sentinel.graph.nodes.answerer import SYSTEM_PROMPT

pytestmark = pytest.mark.unit

# Verbatim reply from groq:openai/gpt-oss-120b to eval case norm-001 (2026-11-14 run).
GPT_OSS_REPLY = "The owner of the Q3 Auth Migration Plan (v2) is Priya Sharma\u30101\u3011."


def test_the_exact_gpt_oss_reply_becomes_a_normal_citation() -> None:
    assert cited_numbers(GPT_OSS_REPLY) == []  # what the parser saw before the fix
    fixed = normalize_markers(GPT_OSS_REPLY)
    assert fixed == "The owner of the Q3 Auth Migration Plan (v2) is Priya Sharma[1]."
    assert cited_numbers(fixed) == [1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Done\uff3b2\uff3d.", "Done[2]."),  # full-width square brackets
        ("Done\u30103\u2020notion:auth-migration-v2\u3011.", "Done[3]."),  # dagger + source
        ("Done\u3010\uff11\u3011.", "Done[1]."),  # full-width digit
        ("Done\u30101, 3\u3011.", "Done[1][3]."),  # a list
        ("Done\u30101\u3011\u30102\u3011.", "Done[1][2]."),  # several markers
        ("Done\u3010 4 \u3011.", "Done[4]."),  # inner spaces
        ("Mixed [1] and \u30102\u3011.", "Mixed [1] and [2]."),
    ],
)
def test_other_marker_styles(raw: str, expected: str) -> None:
    assert normalize_markers(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Plain [1] and [2][3] stay as they are.",
        "A heading \u3010Notice\u3011 without a number is not a citation.",
        "Brackets \uff3bnote\uff3d with words stay.",
        "A very long number \u30101234\u3011 is not a citation marker.",
        "",
    ],
)
def test_text_that_is_not_a_foreign_citation_is_unchanged(text: str) -> None:
    assert normalize_markers(text) == text


def test_the_prompt_requires_ascii_square_brackets() -> None:
    assert "ASCII square brackets" in SYSTEM_PROMPT
    assert "[1] or [1][3]" in SYSTEM_PROMPT
    assert "\u30101\u3011" in SYSTEM_PROMPT  # named as a style not to use
    assert "\uff3b1\uff3d" in SYSTEM_PROMPT
    assert "Never use another citation style" in SYSTEM_PROMPT
