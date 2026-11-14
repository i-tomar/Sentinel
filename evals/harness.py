"""Run eval cases through the real agent graph and turn each run into a `CaseResult`.

Every case gets fresh in-process fixture MCP servers (plus the case's extra records), its
fixture failure rules, a temporary SQLite checkpointer, memory store, approval store and audit
log, and an in-memory span exporter. Nothing is shared between cases, nothing leaves the
machine, and no credential is read: the default model is the deterministic `MockLLM` with the
rule planner. A real model is opt-in through `HarnessConfig` (see evals/run.py).
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
from langchain_core.language_models.chat_models import BaseChatModel
from mcp import Client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from evals.schema import CaseResult, Citation, EvalCase, ExecutedWrite
from sentinel.citations import cited_numbers
from sentinel.graph.build import AskResult, Deps, ask_async, build_graph
from sentinel.graph.checkpoint import open_checkpointer
from sentinel.graph.nodes.execute_tools import MCPToolExecutor
from sentinel.graph.nodes.planner import Planner, RulePlanner
from sentinel.graph.state import Evidence
from sentinel.llm import MockLLM
from sentinel.llm_ratelimit import RateLimitWaits
from sentinel.memory.models import MemoryItem
from sentinel.memory.sqlite_backend import SQLiteMemoryBackend
from sentinel.permissions.approvals import ApprovalStore
from sentinel.permissions.audit import AuditLog
from sentinel.retrieval.embed import HashEmbedder
from sentinel.retrieval.store import ChunkStore
from sentinel.text import split_sentences
from sentinel.tracing import RedactingSpanExporter, get_tracer
from servers.fixtures import failure_modes, gmail_mcp, jira_mcp, notion_mcp
from servers.fixtures.generate import NOW, build_dataset
from servers.fixtures.models import FixtureDataset, GmailThread, JiraIssue, NotionPage

_MARKERS = re.compile(r"\s*\[\d{1,3}\]")


@dataclass
class HarnessConfig:
    llm_factory: Callable[[], BaseChatModel] = MockLLM
    planner_factory: Callable[[BaseChatModel], Planner] = field(
        default_factory=lambda: lambda _llm: RulePlanner()
    )
    tokens_reported: bool = False
    """False for the mock (it reports no real token counts): tokens are left None."""
    model_name: str = "mock"
    tool_timeout_s: float = 2.0
    """Generous for in-process calls (milliseconds locally) so a slow CI runner never turns a
    healthy call into a timeout and changes a deterministic metric."""
    fixture_timeout_s: float = 5.0
    """How long the fixture `timeout` failure mode stalls: well past `tool_timeout_s`."""
    rate_limits: RateLimitWaits | None = None
    """Shared with the model's HTTP clients (Groq): waits are taken out of latency, and an
    exhausted rate limit stops the suite."""
    case_delay_s: float = 0.0
    """Pause between cases, to stay under a provider's per-minute limits. Not latency."""
    case_sleep: Callable[[float], Awaitable[None]] = anyio.sleep


class RateLimitExhaustedError(RuntimeError):
    """The provider kept rate-limiting (or asked for a longer wait than allowed)."""


def agent_latency_ms(elapsed_ms: int, waited_s: float) -> tuple[int, int]:
    """(latency without rate-limit waits, the waits) in ms."""
    waited_ms = round(waited_s * 1000)
    return max(0, elapsed_ms - waited_ms), waited_ms


def corpus_keys(cases: Sequence[EvalCase], dataset: FixtureDataset | None = None) -> set[str]:
    ds = dataset or build_dataset()
    keys = {f"notion:{p.id}" for p in ds.notion}
    keys |= {f"jira:{i.key}" for i in ds.jira}
    keys |= {f"gmail:{t.id}" for t in ds.gmail}
    for case in cases:
        for extra in case.extra_records:
            ident = extra.record.get("key") or extra.record.get("id")
            keys.add(f"{extra.source}:{ident}")
    return keys


@contextmanager
def _fixture_env(rules: str, timeout_s: float) -> Iterator[None]:
    names = (failure_modes.FAILURES_ENV, failure_modes.TIMEOUT_ENV)
    saved = {name: os.environ.get(name) for name in names}
    os.environ[failure_modes.FAILURES_ENV] = rules
    os.environ[failure_modes.TIMEOUT_ENV] = str(timeout_s)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def _no_sleep(_seconds: float) -> None:
    return None


def _claims(answer: str, result: AskResult) -> list[Citation]:
    by_number = {c.n: c.key for c in result.citations}
    claims: list[Citation] = []
    for sentence in split_sentences(answer):
        text = _MARKERS.sub("", sentence).strip()
        for n in cited_numbers(sentence):
            if n in by_number:
                claims.append(Citation(claim=text, key=by_number[n]))
    return claims


def _evidence(values: dict[str, Any]) -> list[Evidence]:
    return [
        e if isinstance(e, Evidence) else Evidence.model_validate(e)
        for e in values.get("evidence", [])
    ]


def _writes(audit: AuditLog) -> list[ExecutedWrite]:
    approved = {r.approval_id for r in audit.records if r.event == "approved"}
    return [
        ExecutedWrite(tool=f"{r.source}.{r.tool}", approved=r.approval_id in approved)
        for r in audit.records
        if r.event == "attempted"
    ]


def _stored(tmp: Path) -> dict[str, str]:
    """Everything the run left on disk, decoded leniently so ASCII values are searchable."""
    return {
        f"file:{p.name}": p.read_bytes().decode("utf-8", errors="replace")
        for p in sorted(tmp.iterdir())
        if p.is_file()
    }


def _records(case: EvalCase, ds: FixtureDataset) -> tuple[list[Any], list[Any], list[Any]]:
    pages: list[Any] = list(ds.notion)
    issues: list[Any] = list(ds.jira)
    threads: list[Any] = list(ds.gmail)
    for extra in case.extra_records:
        if extra.source == "notion":
            pages.append(NotionPage.model_validate(extra.record))
        elif extra.source == "jira":
            issues.append(JiraIssue.model_validate(extra.record))
        else:
            threads.append(GmailThread.model_validate(extra.record))
    return pages, issues, threads


async def run_case(
    case: EvalCase, config: HarnessConfig | None = None, dataset: FixtureDataset | None = None
) -> CaseResult:
    config = config or HarnessConfig()
    ds = dataset or build_dataset()
    try:
        return await _run(case, config, ds)
    except Exception as exc:
        return _expected(case).model_copy(
            update={"answer": "", "harness_error": f"{type(exc).__name__}"}
        )


def _expected(case: EvalCase) -> CaseResult:
    e = case.expected
    return CaseResult(
        case_id=case.id,
        category=case.category,
        surface=case.surface,
        split=case.split,
        question=case.question,
        expected_facts=e.answer_contains,
        forbidden_facts=e.answer_excludes,
        answer_rubric=e.answer_rubric,
        needs_judge=case.needs_judge,
        expected_records=e.expected_records,
        required_citations=e.required_citations,
        forbidden_citations=e.forbidden_citations,
        should_refuse=e.should_refuse,
        attack_goal_strings=e.attack_goal_strings,
        forbidden_tool_calls=e.forbidden_tool_calls,
        write_intent=e.expect_approval_request,
        expect_policy_denial=e.expect_policy_denial,
        seeded_pii=e.seeded_pii,
        answer="",
    )


async def _run(case: EvalCase, config: HarnessConfig, ds: FixtureDataset) -> CaseResult:
    pages, issues, threads = _records(case, ds)
    with (
        tempfile.TemporaryDirectory(prefix="sentinel-eval-", ignore_cleanup_errors=True) as name,
        _fixture_env(case.fixture_overrides, config.fixture_timeout_s),
    ):
        tmp = Path(name)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(RedactingSpanExporter(exporter)))
        audit = AuditLog(tmp / "audit.jsonl")
        approvals = ApprovalStore(tmp / "approvals.sqlite")
        memory = SQLiteMemoryBackend(tmp / "memory.sqlite")
        for seed in case.memory_seeds:
            memory.put(
                MemoryItem(
                    key=seed.key,
                    value=seed.value,
                    source=seed.source,
                    verified=seed.verified,
                    updated_at=NOW - timedelta(days=60),
                )
            )
        llm = config.llm_factory()
        store = ChunkStore(QdrantClient(location=":memory:"), "eval", HashEmbedder())
        store.ensure_collection()
        async with AsyncExitStack() as stack:
            clients = {
                "notion": await stack.enter_async_context(
                    Client(notion_mcp.create_server(pages=pages))
                ),
                "jira": await stack.enter_async_context(
                    Client(jira_mcp.create_server(issues=issues))
                ),
                "gmail": await stack.enter_async_context(
                    Client(gmail_mcp.create_server(threads=threads))
                ),
            }
            saver = await stack.enter_async_context(open_checkpointer(tmp / "checkpoints.sqlite"))
            deps = Deps(
                store=store,
                llm=llm,
                tracer=get_tracer(provider),
                tools=MCPToolExecutor(clients, timeout_s=config.tool_timeout_s),  # type: ignore[arg-type]
                planner=config.planner_factory(llm),
                tool_timeout_s=config.tool_timeout_s,
                sleep=_no_sleep,
                checkpointer=saver,
                memory=memory,
                memory_auto_approve=True,  # the explicit eval flag (fixture data only)
                approvals=approvals,
                audit=audit,
                principal="eval",
            )
            for earlier in case.thread_setup:
                await ask_async(earlier, deps, as_of=NOW, thread_id=case.id)
            waited_before = config.rate_limits.waited_s if config.rate_limits else 0.0
            result = await ask_async(case.question, deps, as_of=NOW, thread_id=case.id)
            waited = (config.rate_limits.waited_s if config.rate_limits else 0.0) - waited_before
            snapshot = await build_graph(deps).aget_state({"configurable": {"thread_id": case.id}})
        approvals.close()
        evidence = _evidence(dict(snapshot.values or {}))
        outputs = {
            "answer": result.answer,
            "result": result.model_dump_json(),
            "spans": "\n".join(s.to_json() for s in exporter.get_finished_spans()),
            "audit": "\n".join(r.model_dump_json() for r in audit.records),
            **_stored(tmp),
        }
        writes = _writes(audit)
        usage = result.token_usage
        latency_ms, wait_ms = agent_latency_ms(result.latency_ms, waited)
        return _expected(case).model_copy(
            update={
                "answer": result.answer,
                "refused": bool(result.input_check and result.input_check.blocked),
                "retrieved": list(result.evidence_keys),
                "returned_records": {e.citation.key: e.text for e in evidence},
                "citations": _claims(result.answer, result),
                "guard_flagged": sorted({n.key for n in result.guard_notes}),
                "tool_calls": [f"{c.source}.{c.tool}" for c in result.tool_calls]
                + [w.tool for w in writes],
                "executed_writes": writes,
                "approval_requested": any(a.status == "pending_approval" for a in result.actions),
                "policy_denied": any(a.status == "policy_denied" for a in result.actions),
                "outputs": outputs,
                "latency_ms": latency_ms,
                "rate_limit_wait_ms": wait_ms,
                "tool_call_count": len(result.tool_calls),
                "input_tokens": usage.input_tokens if config.tokens_reported else None,
                "output_tokens": usage.output_tokens if config.tokens_reported else None,
            }
        )


async def run_suite(
    cases: Sequence[EvalCase],
    config: HarnessConfig | None = None,
    progress: Callable[[int, int, CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Cases run one at a time: failure rules are process-wide environment variables.
    Raises `RateLimitExhaustedError` (after reporting that case) once the provider's rate
    limit could not be waited out, instead of failing every remaining case."""
    config = config or HarnessConfig()
    ds = build_dataset()
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        if i > 1 and config.case_delay_s > 0:
            await config.case_sleep(config.case_delay_s)
        result = await run_case(case, config, ds)
        results.append(result)
        if progress:
            progress(i, len(cases), result)
        if config.rate_limits is not None and config.rate_limits.exhausted:
            raise RateLimitExhaustedError(
                f"stopped after {i}/{len(cases)} cases: {config.rate_limits.exhausted_reason}"
            )
    return results
