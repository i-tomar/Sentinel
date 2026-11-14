"""One evaluated case: what was expected, and what the agent did.

The harness (step p) fills these from dataset rows plus an `AskResult` and the captured
outputs. Metrics only read them, so they can be tested with small hand-made results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "normal",
    "edge",
    "adversarial_user",
    "adversarial_retrieved",
    "tool_failure",
    "memory_freshness",
    "benign_lookalike",
]
Surface = Literal["input", "retrieved"]
"""Where an attack (or an attack-like benign case) arrives: the user's question, or data."""
Split = Literal["dev", "heldout"]
"""dev: may reuse fixture payloads, detector samples or rule wording; heldout: written
independently of both, so the guards were never tuned on it."""


class Expected(BaseModel):
    """Ground truth for one case, written from the fixture records (never from a run)."""

    model_config = ConfigDict(extra="forbid")

    answer_contains: list[str] = Field(default_factory=list)
    answer_excludes: list[str] = Field(default_factory=list)
    answer_rubric: str | None = None
    expected_records: list[str] = Field(default_factory=list)
    required_citations: list[str] = Field(default_factory=list)
    forbidden_citations: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    attack_goal_strings: list[str] = Field(default_factory=list)
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    expect_approval_request: bool = False
    expect_policy_denial: bool = False
    seeded_pii: list[str] = Field(default_factory=list)


class ExtraRecord(BaseModel):
    """A record added to one fixture server for one case only (e.g. a held-out poisoned email
    that is not part of the generated fixture data)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["notion", "jira", "gmail"]
    record: dict[str, Any]


class MemorySeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str
    source: str
    verified: bool = False


class EvalCase(BaseModel):
    """One dataset row (evals/datasets/*.jsonl)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-[a-z0-9-]+$")
    category: Category
    surface: Surface | None = None
    split: Split
    question: str = Field(min_length=1)
    thread_setup: list[str] = Field(default_factory=list)
    """Earlier user questions in the same thread, asked first (follow-ups, multi-turn)."""
    fixture_overrides: str = ""
    """SENTINEL_FIXTURE_FAILURES rules for this case (tool-failure cases)."""
    extra_records: list[ExtraRecord] = Field(default_factory=list)
    memory_seeds: list[MemorySeed] = Field(default_factory=list)
    expected: Expected
    needs_judge: bool = False
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> EvalCase:
        if self.needs_judge and not self.expected.answer_rubric:
            raise ValueError(f"{self.id}: needs_judge requires expected.answer_rubric")
        if self.category.startswith("adversarial") and self.surface is None:
            raise ValueError(f"{self.id}: adversarial cases must name their surface")
        if self.category == "benign_lookalike" and self.surface is None:
            raise ValueError(f"{self.id}: benign lookalikes must name their surface")
        return self


class Citation(BaseModel):
    claim: str
    """The answer sentence that carries the citation marker."""
    key: str
    """The cited record, e.g. "jira:PROJ-142"."""


class ExecutedWrite(BaseModel):
    tool: str
    """"source.tool", e.g. "jira.update_issue"."""
    approved: bool
    """True only if a user approval for exactly this call was consumed."""


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: Category
    surface: Surface | None = None
    split: Split = "dev"
    question: str = ""

    # --- expected (from the dataset) ----------------------------------------------------
    expected_facts: list[str] = Field(default_factory=list)
    """Strings the answer must contain (case-, width- and whitespace-insensitive)."""
    forbidden_facts: list[str] = Field(default_factory=list)
    """Strings the answer must not contain (e.g. a stale value that memory still holds)."""
    answer_rubric: str | None = None
    """For the optional LLM judge; never used by the deterministic check."""
    needs_judge: bool = False
    """Correctness can only be judged against the rubric (negation, paraphrase)."""
    expect_policy_denial: bool = False
    """The requested action must be refused by policy without an approval request."""
    expected_records: list[str] = Field(default_factory=list)
    """Record keys a good retrieval returns (recall@k)."""
    required_citations: list[str] = Field(default_factory=list)
    forbidden_citations: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    attack_goal_strings: list[str] = Field(default_factory=list)
    """Text that only appears if the attack worked (an attacker address, "PWNED-17")."""
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    write_intent: bool = False
    """The user explicitly asked for a write: an approval request is expected."""
    seeded_pii: list[str] = Field(default_factory=list)
    """PII values that must not appear in any output path for this case."""

    # --- observed (from the run) --------------------------------------------------------
    answer: str
    refused: bool = False
    """The agent declined (input guard block, or a refusal reply)."""
    retrieved: list[str] = Field(default_factory=list)
    """Record keys returned by tools or the index, in rank order (evidence order)."""
    returned_records: dict[str, str] = Field(default_factory=dict)
    """Text of every record actually returned this turn, by key (for citation support)."""
    citations: list[Citation] = Field(default_factory=list)
    guard_flagged: list[str] = Field(default_factory=list)
    """Record keys the retrieved-content guard neutralized something in."""
    tool_calls: list[str] = Field(default_factory=list)
    """Every tool call attempted, as "source.tool"."""
    executed_writes: list[ExecutedWrite] = Field(default_factory=list)
    approval_requested: bool = False
    policy_denied: bool = False
    outputs: dict[str, str] = Field(default_factory=dict)
    """Everything that left the agent, by path: answer, result JSON, spans, audit, stores."""
    latency_ms: int = 0
    """End-to-end agent time, with provider rate-limit waits taken out."""
    rate_limit_wait_ms: int = 0
    """Time spent waiting on provider rate limits (HTTP 429) during the case; not latency."""
    tool_call_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    """None means the provider did not report tokens (e.g. the mock); never a made-up 0."""
    harness_error: str | None = None
    """The case could not run (exception type only); every expectation then counts as failed."""
