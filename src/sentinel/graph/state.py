"""Typed agent state. Later steps add sub-questions, hops, memory, approvals."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sentinel.citations import Citation
from sentinel.retrieval.models import SOURCES, Source


class Evidence(BaseModel):
    """One retrieved record: its citation plus the matched chunk text (untrusted data)."""

    citation: Citation
    text: str
    score: float


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


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
    candidates: int = 0
    """Chunks returned by search before the relevance gate."""
    evidence: list[Evidence] = Field(default_factory=list)
    draft: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    citation_report: CitationReport = Field(default_factory=CitationReport)
    insufficient_evidence: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[str] = Field(default_factory=list)
