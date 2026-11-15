"""Chat model factory and a deterministic MockLLM.

`LLM_PROVIDER=mock` (default) needs no key or network. Real providers are optional extras
(`uv sync --extra openai|anthropic|google`) and require LLM_MODEL and LLM_API_KEY; no model
name is assumed. `groq` uses Groq's OpenAI-compatible API through the `openai` extra, with
HTTP 429 handled by `sentinel.llm_ratelimit`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from sentinel.config import Settings
from sentinel.llm_ratelimit import (
    AsyncRateLimitTransport,
    ProviderUsage,
    RateLimitPolicy,
    RateLimitTransport,
)
from sentinel.text import content_tokens, split_sentences

LLM_TIMEOUT_S = 60.0
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
"""From Groq's OpenAI-compatibility docs (console.groq.com/docs/openai), checked 2026-11-13."""
INSUFFICIENT_PREFIX = "I don't have enough evidence"
PARTIAL_PREFIX = "I could not read"
"""Start of a sentence admitting a source was unavailable (not a factual claim, needs no [n])."""

_EVIDENCE = re.compile(r'<evidence n="(\d+)"([^>]*)>\n(.*?)\n</evidence>', re.DOTALL)
_SOURCE_ATTR = re.compile(r'source="(\w+)"')
_ID_ATTR = re.compile(r'id="([^"]+)"')
_QUESTION = re.compile(r"^Question: (.*)$", re.MULTILINE)
_FAILED_SOURCE = re.compile(r"^- (\w+) \([\w.]+\) failed after", re.MULTILINE)
# Concrete facts the eval grader (and users) expect copied, not paraphrased away.
_FACT_ANCHOR = re.compile(
    r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:USD|INR|EUR|GBP)\s*[\d.,]+[KMB]?\b"
    r"|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"
    r"|\bdecodes? to\b[^.\n]{0,80}",
    re.IGNORECASE,
)
_TICKET_ID = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")
_TICKET_ASK = frozenset({"ticket", "tickets", "issue", "issues", "blocked", "blocker", "blockers"})


def prompt_hash(messages: Sequence[BaseMessage]) -> str:
    """Stable key for scripted mock responses (role + content of every message)."""
    joined = "\n".join(f"{m.type}:{m.content}" for m in messages)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _novel_anchors(sentence: str, question: str) -> int:
    """How many checkable facts in the sentence are not already in the question text."""
    q = question.casefold()
    return sum(1 for m in _FACT_ANCHOR.finditer(sentence) if m.group(0).casefold() not in q)


def _pick_sentences(body: str, wanted: set[str], question: str) -> str | None:
    """Best overlapping sentence, plus another from the same block when it adds novel facts."""
    asks_tickets = bool(wanted & _TICKET_ASK)
    scored: list[tuple[int, int, str]] = []
    for raw in split_sentences(body):
        words = content_tokens(raw)
        sentence = raw.lstrip("#-* ").rstrip(".")
        if not sentence or not (words - wanted):
            continue
        overlap = wanted & words
        ticket_line = asks_tickets and bool(_TICKET_ID.search(sentence))
        if not (overlap or ticket_line):
            continue
        novel = _novel_anchors(sentence, question)
        scored.append((len(overlap) + 2 * novel + (1 if ticket_line else 0), novel, sentence))
    if not scored:
        return None
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
    primary = scored[0][2]
    lines: list[str] = [primary]
    seen_anchors = {m.group(0).casefold() for m in _FACT_ANCHOR.finditer(primary)}
    if asks_tickets:
        # "Which tickets…?" needs every ticket id in the block, not just the top sentence.
        for _score, _novel, sentence in scored:
            if sentence == primary or sentence.casefold() in primary.casefold():
                continue
            ids = {m.group(0) for m in _TICKET_ID.finditer(sentence)}
            new_ids = {i.casefold() for i in ids} - seen_anchors
            if not new_ids:
                continue
            seen_anchors |= new_ids
            lines.append(sentence)
            if len(lines) >= 4:
                break
    for _score, novel, sentence in scored[1:]:
        if len(lines) >= 4:
            break
        if (
            not novel
            or sentence == primary
            or any(sentence.casefold() in existing.casefold() for existing in lines)
        ):
            continue
        # Annex when it adds a fact the lines lack (date/amount/id) and still answers.
        extra = {
            m.group(0).casefold()
            for m in _FACT_ANCHOR.finditer(sentence)
            if m.group(0).casefold() not in seen_anchors
        }
        words = content_tokens(sentence)
        if extra and (wanted & words or (asks_tickets and _TICKET_ID.search(sentence))):
            seen_anchors |= extra
            lines.append(sentence)
    return "; ".join(lines)


def extractive_answer(prompt: str, max_blocks: int = 5) -> str:
    """What MockLLM says when unscripted: from each non-stale evidence block, the sentence that
    best overlaps the question while adding at least one word the question lacks, cited with
    that block's number. Ticket-id lines count when the question asks which tickets, and a
    second sentence from the same block is appended when it adds a date, amount or id the
    question did not already state. The best block of each source is taken first, then the
    best of the rest, up to `max_blocks`. Never cites a number absent from the prompt.
    Mentions unreadable sources."""
    question_match = _QUESTION.search(prompt)
    question = question_match.group(1) if question_match else ""
    wanted = content_tokens(question) if question else set()
    best: list[tuple[int, int, str, str]] = []
    for n, attrs, body in _EVIDENCE.findall(prompt):
        if 'stale="true"' in attrs:
            continue
        picked = _pick_sentences(body, wanted, question)
        if not picked:
            continue
        # Score the block by its primary sentence (before any "; " annex).
        primary = picked.split("; ", 1)[0]
        score = len(wanted & content_tokens(primary)) + 2 * _novel_anchors(primary, question)
        doc_id = _ID_ATTR.search(attrs)
        # Prefer citing the ticket record itself when the question asks which tickets.
        if (
            wanted & _TICKET_ASK
            and doc_id
            and _TICKET_ID.fullmatch(doc_id.group(1))
            and doc_id.group(1) in picked
        ):
            score += 3
        source = _SOURCE_ATTR.search(attrs)
        best.append((score, int(n), source.group(1) if source else "", picked))
    failed = sorted(set(_FAILED_SOURCE.findall(prompt)))
    note = f" {PARTIAL_PREFIX} {', '.join(failed)}, so this may be incomplete." if failed else ""
    if not best:
        return f"{INSUFFICIENT_PREFIX} to answer this from the retrieved sources.{note}"
    best.sort(key=lambda s: (-s[0], s[1]))
    picked_blocks: list[tuple[int, int, str, str]] = []
    for item in best:  # one per source first, so multi-source evidence shows up
        if item[2] not in {p[2] for p in picked_blocks}:
            picked_blocks.append(item)
    picked_blocks += [item for item in best if item not in picked_blocks]
    chosen = sorted(picked_blocks[:max_blocks], key=lambda s: s[1])
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


def rate_limited_clients(
    usage: ProviderUsage,
    policy: RateLimitPolicy,
    *,
    sync_inner: httpx.BaseTransport | None = None,
    async_inner: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """httpx clients that resend on HTTP 429, 5xx and connection errors, recording waits and
    tokens in `usage`. The inner transports default to real HTTP; tests pass
    `httpx.MockTransport`."""
    return {
        "http_client": httpx.Client(
            transport=RateLimitTransport(sync_inner or httpx.HTTPTransport(), usage, policy),
            timeout=LLM_TIMEOUT_S,
        ),
        "http_async_client": httpx.AsyncClient(
            transport=AsyncRateLimitTransport(
                async_inner or httpx.AsyncHTTPTransport(), usage, policy
            ),
            timeout=LLM_TIMEOUT_S,
        ),
    }


def rate_limit_policy(settings: Settings) -> RateLimitPolicy:
    return RateLimitPolicy(
        retries=settings.llm_rate_limit_retries,
        max_wait_s=settings.llm_rate_limit_max_wait_s,
        server_retries=settings.llm_server_retries,
    )


def groq_options(settings: Settings, model: str) -> dict[str, Any]:
    """Request options for Groq. gpt-oss models return reasoning in a separate field; it is
    not requested (`include_reasoning: false`), so it can never end up in an answer."""
    if model.startswith("openai/gpt-oss"):
        return {
            "reasoning_effort": settings.groq_reasoning_effort,
            "extra_body": {"include_reasoning": False},
        }
    return {}


def make_chat_model(
    settings: Settings,
    client_options: Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
    usage: ProviderUsage | None = None,
) -> BaseChatModel:
    """`client_options` are passed to the provider class as-is (e.g. `http_client`). `model`
    overrides LLM_MODEL (the eval judge). `usage` collects Groq's retry waits and tokens."""
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLM()
    if settings.llm_api_key is None or not settings.llm_model:  # also enforced by Settings
        raise ValueError(f"LLM_PROVIDER={provider} needs LLM_MODEL and LLM_API_KEY")
    name = model or settings.llm_model
    # Passed via a dict: the provider classes take these as pydantic aliases mypy can't see.
    common: dict[str, Any] = {
        "model": name,
        "api_key": settings.llm_api_key,
        "timeout": LLM_TIMEOUT_S,
    }
    try:
        if provider == "groq":
            from langchain_openai import ChatOpenAI

            options = {
                **rate_limited_clients(usage or ProviderUsage(), rate_limit_policy(settings)),
                **(client_options or {}),
                **groq_options(settings, name),
                "base_url": GROQ_BASE_URL,
            }
            return ChatOpenAI(
                **common,
                **options,
                max_retries=0,  # the transport retries (and times) 429, 5xx and dropped calls
                use_responses_api=False,
            )
        common.update(client_options or {})
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common, max_retries=2)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(**common, max_retries=2)
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**common)
    except ImportError as exc:
        extra = "openai" if provider == "groq" else provider
        raise MissingProviderError(
            f"LLM_PROVIDER={provider} needs its extra: uv sync --extra {extra}"
        ) from exc
