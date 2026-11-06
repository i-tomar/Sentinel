"""Chat model factory and a deterministic MockLLM.

`LLM_PROVIDER=mock` (default) needs no key or network. Real providers are optional extras
(`uv sync --extra openai|anthropic|google`) and require LLM_MODEL and LLM_API_KEY; no model
name is assumed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from sentinel.config import Settings
from sentinel.text import content_tokens, split_sentences

LLM_TIMEOUT_S = 60.0
INSUFFICIENT_PREFIX = "I don't have enough evidence"
PARTIAL_PREFIX = "I could not read"
"""Start of a sentence admitting a source was unavailable (not a factual claim, needs no [n])."""

_EVIDENCE = re.compile(r'<evidence n="(\d+)"([^>]*)>\n(.*?)\n</evidence>', re.DOTALL)
_SOURCE_ATTR = re.compile(r'source="(\w+)"')
_QUESTION = re.compile(r"^Question: (.*)$", re.MULTILINE)
_FAILED_SOURCE = re.compile(r"^- (\w+) \([\w.]+\) failed after", re.MULTILINE)


def prompt_hash(messages: Sequence[BaseMessage]) -> str:
    """Stable key for scripted mock responses (role + content of every message)."""
    joined = "\n".join(f"{m.type}:{m.content}" for m in messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def extractive_answer(prompt: str, max_blocks: int = 5) -> str:
    """What MockLLM says when unscripted: from each non-stale evidence block, the sentence that
    best overlaps the question while adding at least one word the question lacks, cited with
    that block's number. The best block of each source is taken first, then the best of the
    rest, up to `max_blocks`. Never cites a number absent from the prompt. Mentions
    unreadable sources."""
    question = _QUESTION.search(prompt)
    wanted = content_tokens(question.group(1)) if question else set()
    best: list[tuple[int, int, str, str]] = []
    for n, attrs, body in _EVIDENCE.findall(prompt):
        if 'stale="true"' in attrs:
            continue
        scored = []
        for raw in split_sentences(body):
            words = content_tokens(raw)
            sentence = raw.lstrip("#-* ").rstrip(".")
            if sentence and wanted & words and words - wanted:
                scored.append((len(wanted & words), sentence))
        if scored:
            overlap, sentence = max(scored, key=lambda s: s[0])
            source = _SOURCE_ATTR.search(attrs)
            best.append((overlap, int(n), source.group(1) if source else "", sentence))
    failed = sorted(set(_FAILED_SOURCE.findall(prompt)))
    note = f" {PARTIAL_PREFIX} {', '.join(failed)}, so this may be incomplete." if failed else ""
    if not best:
        return f"{INSUFFICIENT_PREFIX} to answer this from the retrieved sources.{note}"
    best.sort(key=lambda s: (-s[0], s[1]))
    picked: list[tuple[int, int, str, str]] = []
    for item in best:  # one per source first, so multi-source evidence shows up
        if item[2] not in {p[2] for p in picked}:
            picked.append(item)
    picked += [item for item in best if item not in picked]
    chosen = sorted(picked[:max_blocks], key=lambda s: s[1])
    return " ".join(f"{sentence} [{n}]." for _, n, _, sentence in chosen) + note


class MockLLM(BaseChatModel):
    """Deterministic offline chat model. Scripted replies are keyed by `prompt_hash`."""

    scripted: dict[str, str] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)
    """Prompt hashes seen, in order (lets tests assert whether the model was called)."""

    @property
    def _llm_type(self) -> str:
        return "sentinel-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        key = prompt_hash(messages)
        self.calls.append(key)
        text = self.scripted.get(key) or extractive_answer(str(messages[-1].content))
        # Token counts are reported as 0: the mock does not tokenize, and evals must not show
        # invented numbers.
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        message = AIMessage(content=text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


class MissingProviderError(RuntimeError):
    pass


def make_chat_model(
    settings: Settings, client_options: Mapping[str, Any] | None = None
) -> BaseChatModel:
    """`client_options` are passed to the provider class as-is (e.g. `http_client`)."""
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLM()
    if settings.llm_api_key is None or not settings.llm_model:  # also enforced by Settings
        raise ValueError(f"LLM_PROVIDER={provider} needs LLM_MODEL and LLM_API_KEY")
    # Passed via a dict: the provider classes take these as pydantic aliases mypy can't see.
    common: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "timeout": LLM_TIMEOUT_S,
        **(client_options or {}),
    }
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common, max_retries=2)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common, max_retries=2)
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**common)
    except ImportError as exc:
        raise MissingProviderError(
            f"LLM_PROVIDER={provider} needs its extra: uv sync --extra {provider}"
        ) from exc
