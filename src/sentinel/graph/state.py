"""Typed agent state. Later steps add memory and approvals."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.citations import Citation
from sentinel.retrieval.models import SOURCES, Chunk, Document, Source

Relation = Literal["search", "blocked_by", "mentions"]
"""search: keyword search in the target sources.
blocked_by: Jira issues blocked by ticket keys found in the dependency's records.
mentions: email threads mentioning Jira tickets found by the dependency."""

TimeWindow = Literal["last_week", "this_week", "last_7_days"]


class SubQuestion(BaseModel):
    """One step of the plan. Derived from the user question only, never from retrieved data."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^sq\d{1,2}$")
    text: str = Field(min_length=1, max_length=300)
    target_sources: list[Source] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    relation: Relation = "search"
    query: str = Field(default="", max_length=200)
    time_window: TimeWindow | None = None
    label: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,30}$")


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
            if sq.relation != "search" and not sq.depends_on:
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
