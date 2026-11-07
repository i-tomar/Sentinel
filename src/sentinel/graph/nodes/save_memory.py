"""Propose long-term memory writes after an answer and send them through the permission gate.

Two kinds of write:
- mark_stale for notes that conflicted with a newer source record this turn;
- put of the answer as a verified fact, only when every sentence passed the citation check
  against records fetched this turn (no unknown, uncited, unsupported or memory-only claims).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from sentinel.citations import Citation, cited_numbers
from sentinel.graph.nodes.answerer import is_insufficient
from sentinel.graph.state import CitationReport, Evidence
from sentinel.llm import INSUFFICIENT_PREFIX, PARTIAL_PREFIX
from sentinel.memory.backend import MemoryBackend
from sentinel.memory.freshness import support
from sentinel.memory.models import MemoryItem, MemoryWrite, memory_key
from sentinel.memory.policy import WriteDecision
from sentinel.permissions.actions import Action
from sentinel.permissions.approvals import ApprovalStore
from sentinel.permissions.audit import AuditLog
from sentinel.permissions.gate import ActionRunner, approve_action, request_action
from sentinel.permissions.policy import DataMode, Policy
from sentinel.text import split_sentences

MAX_VALUE_CHARS = 500
AUTO_APPROVE_REASON = "auto_approve_memory (fixture data only)"
_MARKERS = re.compile(r"\s*\[\d{1,3}\]")


def fact_from_answer(
    question: str,
    answer: str,
    citations: Sequence[Citation],
    report: CitationReport,
    evidence: Sequence[Evidence],
) -> MemoryItem | None:
    if is_insufficient(answer) or not citations or not report.ok:
        return None
    sentences = [
        s
        for s in split_sentences(answer)
        if cited_numbers(s) and not s.startswith((PARTIAL_PREFIX, INSUFFICIENT_PREFIX))
    ]
    if not sentences:
        return None
    value = " ".join(_MARKERS.sub("", s) for s in sentences)
    value = " ".join(value.split())[:MAX_VALUE_CHARS].strip()
    if not value:
        return None
    by_key = {ev.citation.key: ev for ev in evidence}
    primary = citations[0]
    cited_text = "\n".join(by_key[c.key].text for c in citations if c.key in by_key)
    return MemoryItem(
        key=memory_key(question),
        value=value,
        source=primary.key,
        verified=True,
        stale=False,
        updated_at=primary.updated_at,
        confidence=round(min(1.0, support(value, cited_text)), 3),
        record_keys=[c.key for c in citations][:8],
    )


def propose_writes(
    conflicting: Sequence[MemoryItem], fact: MemoryItem | None, existing: MemoryItem | None
) -> list[MemoryWrite]:
    writes = [
        MemoryWrite(
            op="mark_stale",
            key=n.key,
            status="pending_approval",
            reason=n.stale_reason or "conflicts with a newer source record",
            item=n,
        )
        for n in conflicting
    ]
    if fact is not None:
        unchanged = (
            existing is not None
            and not existing.stale
            and existing.value == fact.value
            and existing.source == fact.source
        )
        writes.append(
            MemoryWrite(
                op="put",
                key=fact.key,
                status="skipped" if unchanged else "pending_approval",
                reason="already remembered" if unchanged else "verified answer from cited records",
                item=fact,
            )
        )
    return writes


def memory_action(write: MemoryWrite) -> Action | None:
    """The memory tool call a proposed write stands for (what the approval is bound to)."""
    if write.item is None:
        return None
    if write.op == "put":
        return Action(source="memory", tool="put_memory", args=write.item.model_dump(mode="json"))
    return Action(
        source="memory",
        tool="mark_stale",
        args={
            "key": write.key,
            "reason": write.reason[:300],
            "superseded_by": write.item.superseded_by,
        },
    )


async def gate_writes(
    writes: Sequence[MemoryWrite],
    *,
    backend: MemoryBackend | None,
    decision: WriteDecision,
    policy: Policy,
    data_mode: DataMode,
    store: ApprovalStore,
    audit: AuditLog,
    session: str,
    principal: str,
) -> list[MemoryWrite]:
    """Open an approval request for every proposed write. With the fixture-only auto-approve
    flag (`decision == "allow"`) the request is approved on the user's behalf and run at
    once, still through the gate and the audit log; otherwise it waits for `sentinel approve`."""
    runner = ActionRunner(
        store=store, audit=audit, policy=policy, data_mode=data_mode, memory=backend
    )
    out: list[MemoryWrite] = []
    for write in writes:
        action = memory_action(write)
        if write.status != "pending_approval" or backend is None or action is None:
            out.append(write)
            continue
        record = request_action(
            action,
            policy=policy,
            data_mode=data_mode,
            store=store,
            audit=audit,
            session=session,
            principal=principal,
        )
        if record.approval_id is None:
            out.append(write.model_copy(update={"status": "skipped", "reason": record.reason}))
            continue
        write = write.model_copy(update={"approval_id": record.approval_id})
        if decision == "allow":
            approve_action(
                record.approval_id,
                store=store,
                audit=audit,
                session=session,
                principal=principal,
                via="auto_approve_memory",
                reason=AUTO_APPROVE_REASON,
            )
            outcome = await runner.execute(record.approval_id, session=session, principal=principal)
            status = "written" if outcome.status == "succeeded" else "failed"
            write = write.model_copy(update={"status": status})
        out.append(write)
    return out


def flag_conflicts(
    notes: Sequence[MemoryItem], conflicts: Mapping[str, tuple[str, str | None]]
) -> list[MemoryItem]:
    """Notes with this turn's conflicts applied (stale, reason, superseding record)."""
    return [
        n.model_copy(
            update={
                "stale": True,
                "stale_reason": conflicts[n.key][0],
                "superseded_by": conflicts[n.key][1],
            }
        )
        if n.key in conflicts
        else n
        for n in notes
    ]
