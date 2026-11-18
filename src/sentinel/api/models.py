"""Request/response models for the demo API. Extra client fields are rejected."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sentinel.citations import Citation
from sentinel.graph.state import CitationReport, GuardNote, InputCheck
from sentinel.permissions.actions import ActionRecord

MAX_QUESTION_CHARS = 2000
RunStatus = Literal["refused", "completed", "citation_failed", "output_redacted"]
StageStatus = Literal["started", "completed", "failed"]


class CreateRunRequest(BaseModel):
    """Only question and optional thread_id. Provider/model/tools overrides are not accepted."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    thread_id: str | None = Field(default=None, max_length=128)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must not be blank")
        return text


class DataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixture"] = "fixture"
    llm: Literal["mock"] = "mock"
    is_fixture: Literal[True] = True


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    llm: Literal["mock"] = "mock"
    data_mode: Literal["fixture"] = "fixture"
    checked_at: datetime


class CreateRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    mode: Literal["fixture"] = "fixture"


class RunResultResponse(BaseModel):
    """AskResult-shaped payload safe for a UI: no tool args, no secrets, provenance fixed."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    question: str
    answer: str
    citations: list[Citation]
    input_check: InputCheck | None
    guard_notes: list[GuardNote]
    citation_report: CitationReport
    actions: list[ActionRecord]
    output_redactions: list[str]
    session_id: str
    trace_id: str
    latency_ms: int
    thread_id: str | None = None
    data_provenance: DataProvenance = Field(default_factory=DataProvenance)


class StageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    status: StageStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    summary: str


class RunEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    events: list[StageEvent]


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


def run_status_from_result(
    *,
    blocked: bool,
    citation_ok: bool,
    output_redactions: list[str],
) -> RunStatus:
    if blocked:
        return "refused"
    if not citation_ok:
        return "citation_failed"
    if output_redactions:
        return "output_redacted"
    return "completed"


def action_records_public(actions: list[ActionRecord]) -> list[ActionRecord]:
    """ActionRecord already omits raw args; return a copy for response isolation."""
    return [a.model_copy(deep=True) for a in actions]


def safe_json(value: Any) -> Any:
    """JSON-ready dump without raising on unexpected types."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
