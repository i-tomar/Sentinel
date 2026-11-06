"""Typed agent state. Short-term memory (per thread) lives in `turns` and `documents`."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.citations import Citation
from sentinel.retrieval.models import SOURCES, Chunk, Document, Source

Relation = Literal["search", "blocked_by", "mentions", "recall"]
"""search: keyword search in the target sources.
blocked_by: Jira issues blocked by ticket keys found in the dependency's records.
mentions: email threads mentioning Jira tickets found by the dependency.
recall: answer from records already fetched earlier in this thread (no tool calls)."""

TimeWindow = Literal["last_week", "this_week", "last_7_days"]
RECORD_KEY_PATTERN = r"^(notion|jira|gmail):[\w.-]{1,100}$"


class SubQuestion(BaseModel):
    """One step of the plan. Derived from the user's questions only, never from retrieved
    text; `prior_keys` are opaque ids of records this thread already fetched."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^sq\d{1,2}$")
    text: str = Field(min_length=1, max_length=300)
    target_sources: list[Source] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    prior_keys: list[str] = Field(default_factory=list, max_length=12)
    """Records from earlier turns this sub-question builds on (follow-ups)."""
    relation: Relation = "search"
    query: str = Field(default="", max_length=200)
    time_window: TimeWindow | None = None
    label: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,30}$")

    @model_validator(mode="after")
    def _prior_keys_are_record_keys(self) -> SubQuestion:
        bad = [k for k in self.prior_keys if not re.match(RECORD_KEY_PATTERN, k)]
        if bad:
            raise ValueError(f"invalid record keys {bad}")
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _dependencies_point_backwards(self) -> Plan:
        seen: set[str] = set()
        for sq in self.sub_questions:
            if sq.id in seen:
                raise ValueError(f"duplicate sub-question id {sq.id}")
            unknown = [d for d in sq.depends_on if d not in seen]
            if unknown:
                raise ValueError(f"{sq.id} depends on unknown or later ids {unknown}")
            if sq.relation == "recall" and (sq.depends_on or not sq.prior_keys):
                raise ValueError(f"{sq.id}: recall needs prior_keys and no depends_on")
            if sq.relation in ("blocked_by", "mentions") and not (sq.depends_on or sq.prior_keys):
                raise ValueError(f"{sq.id}: relation {sq.relation} needs a dependency")
            seen.add(sq.id)
        return self


class ToolCall(BaseModel):
    id: str
    sub_question_id: str
    hop: int
    source: Source
    tool: str
    args: dict[str, str | int]
    expand: int = 3
    """How many top search results to fetch in full with the source's get tool."""


class ToolFailure(BaseModel):
    call_id: str
    sub_question_id: str
    source: Source
    tool: str
    error: str
    attempts: int


class Evidence(BaseModel):
    """One retrieved record: its citation plus its text (untrusted data)."""

    citation: Citation
    text: str
    score: float
    stale: bool = False
    """Superseded by a newer record on the same topic."""


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class UnsupportedClaim(BaseModel):
    sentence: str
    cited: list[int]
    support: float
    """Share of the sentence's content words found in the cited evidence."""


class CitationReport(BaseModel):
    unknown: list[int] = Field(default_factory=list)
    """Markers that matched no retrieved evidence; removed from the final answer."""
    uncited_sentences: list[str] = Field(default_factory=list)
    unsupported: list[UnsupportedClaim] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unknown or self.uncited_sentences or self.unsupported)


class TurnRecord(BaseModel):
    """What a finished turn leaves in the thread's short-term memory."""

    question: str
    answer: str
    as_of: datetime
    plan: list[SubQuestion]
    unanswered: list[str]
    sq_docs: dict[str, list[str]]
    evidence_keys: list[str]
    cited_keys: list[str]
    failed_sources: list[Source]


class AgentState(BaseModel):
    question: str
    sources: list[Source] = Field(default_factory=lambda: list(SOURCES))
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """"Now" for relative dates such as "last week"."""

    # planning and the hop loop
    plan: list[SubQuestion] = Field(default_factory=list)
    hops: int = 0
    current: list[str] = Field(default_factory=list)
    """Sub-question ids routed in the current hop."""
    done: list[str] = Field(default_factory=list)
    widened: list[str] = Field(default_factory=list)
    """Search sub-questions retried across all allowed sources after finding nothing."""
    next_step: Literal["route", "answer"] = "route"
    unanswered: list[str] = Field(default_factory=list)

    # tools
    pending_calls: list[ToolCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    failures: list[ToolFailure] = Field(default_factory=list)
    documents: dict[str, Document] = Field(default_factory=dict)
    sq_docs: dict[str, list[str]] = Field(default_factory=dict)
    """Sub-question id -> keys of records its tool calls returned."""

    # index retrieval
    candidates: int = 0
    """Index chunks returned by search before the relevance gate."""
    index_chunks: list[Chunk] = Field(default_factory=list)
    stale_keys: list[str] = Field(default_factory=list)

    # answer
    evidence: list[Evidence] = Field(default_factory=list)
    draft: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    citation_report: CitationReport = Field(default_factory=CitationReport)
    insufficient_evidence: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[str] = Field(default_factory=list)

    # short-term memory: survives turns of the same thread via the checkpointer
    turns: list[TurnRecord] = Field(default_factory=list)


MEMORY_FIELDS = frozenset({"documents", "turns"})
"""Fields carried across turns of a thread. Every other field is reset for each new turn."""


def new_turn_input(question: str, sources: list[Source], as_of: datetime) -> dict[str, Any]:
    """Graph input for a new turn: fresh per-turn fields, memory fields omitted so the
    checkpointer keeps the thread's previous values (or the defaults for a new thread)."""
    fresh = AgentState(question=question, sources=sources, as_of=as_of)
    return {
        name: getattr(fresh, name) for name in AgentState.model_fields if name not in MEMORY_FIELDS
    }
