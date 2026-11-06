from __future__ import annotations

import pytest

from sentinel.text import content_tokens, split_sentences, stem

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("word", "expected"),
    [("owns", "own"), ("owner", "own"), ("blocked", "block"), ("tickets", "ticket"), ("is", "is")],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_content_tokens_drop_stopwords_keep_ids() -> None:
    assert content_tokens("Who owns the PROJ-142 ticket?") == {
        "own",
        "proj-142",
        "proj",
        "142",
        "ticket",
    }


def test_split_sentences() -> None:
    assert split_sentences("One [1]. Two? Three!\nFour") == ["One [1].", "Two?", "Three!", "Four"]
