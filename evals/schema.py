"""One evaluated case: what was expected, and what the agent did.

The harness (step p) fills these from dataset rows plus an `AskResult` and the captured
outputs. Metrics only read them, so they can be tested with small hand-made results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    question: str = ""

    # --- expected (from the dataset) ----------------------------------------------------
    expected_facts: list[str] = Field(default_factory=list)
    """Strings the answer must contain (case-, width- and whitespace-insensitive)."""
    answer_rubric: str | None = None
    """For the optional LLM judge; never used by the deterministic check."""
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
    outputs: dict[str, str] = Field(default_factory=dict)
    """Everything that left the agent, by path: answer, result JSON, spans, audit, stores."""
    latency_ms: int = 0
    tool_call_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    """None means the provider did not report tokens (e.g. the mock); never a made-up 0."""
