"""The permission gate: request approval for a proposed action, record the user's decision,
and run an approved action exactly once.

Proposing never executes. Execution re-checks the policy, consumes the approval (bound to
session, user and argument fingerprint) and passes the resulting grant down to `call_json`,
which verifies it again. Writes are not retried: a failed write needs a fresh approval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import anyio
from mcp import Client
from pydantic import BaseModel, ValidationError

from sentinel.mcp_clients.registry import summarize
from sentinel.mcp_clients.tools import ToolCallError, call_json
from sentinel.memory.backend import MemoryBackend
from sentinel.memory.models import MemoryItem
from sentinel.permissions.actions import Action, ActionRecord, Grant, PermissionDenied
from sentinel.permissions.approvals import Approval, ApprovalError, ApprovalStore
from sentinel.permissions.audit import AuditLog
from sentinel.permissions.policy import DataMode, Policy
from sentinel.retrieval.models import Source


class ActionOutcome(BaseModel):
    approval_id: str
    status: Literal["succeeded", "failed", "blocked"]
    reason: str | None = None
    result: dict[str, Any] | None = None


def request_action(
    action: Action,
    *,
    policy: Policy,
    data_mode: DataMode,
    store: ApprovalStore,
    audit: AuditLog,
    session: str,
    principal: str,
) -> ActionRecord:
    """Evaluate the policy and open an approval request. Nothing is executed."""
    decision = policy.evaluate(action, data_mode)
    summary = summarize(action.source, action.tool, action.args)
    if decision.verdict != "require_approval":
        reason = decision.reason if decision.verdict == "deny" else "reads need no approval"
        audit.record("policy_denied", action, session=session, principal=principal, reason=reason)
        return ActionRecord(
            approval_id=None,
            source=action.source,
            tool=action.tool,
            risk=decision.risk,
            summary=summary,
            status="policy_denied",
            reason=reason,
        )
    risk = decision.risk or "write"
    approval = store.request(
        action,
        risk=risk,
        session=session,
        principal=principal,
        confirmation=decision.confirmation,
    )
    audit.record(
        "proposed",
        action,
        session=session,
        principal=principal,
        approval_id=approval.id,
        reason=decision.reason,
    )
    return ActionRecord(
        approval_id=approval.id,
        source=action.source,
        tool=action.tool,
        risk=decision.risk,
        summary=summary,
        status="pending_approval",
        reason=decision.reason,
        confirmation=decision.confirmation,
    )


def _action_of(approval: Approval) -> Action:
    return Action(source=approval.source, tool=approval.tool, args=approval.args or {})


def approve_action(
    approval_id: str,
    *,
    store: ApprovalStore,
    audit: AuditLog,
    session: str,
    principal: str,
    confirmation: str | None = None,
    via: str = "user",
    reason: str | None = None,
) -> Approval:
    """Record the user's approval. Raises ApprovalError (and audits it) when refused."""
    try:
        approval = store.approve(
            approval_id, session=session, principal=principal, confirmation=confirmation, via=via
        )
    except ApprovalError as exc:
        _audit_refusal(store, audit, approval_id, session, principal, exc.reason)
        raise
    audit.record(
        "approved",
        _action_of(approval),
        session=session,
        principal=principal,
        approval_id=approval_id,
        reason=reason,
    )
    return approval


def deny_action(
    approval_id: str, *, store: ApprovalStore, audit: AuditLog, session: str, principal: str
) -> Approval:
    before = store.get(approval_id)
    try:
        approval = store.deny(approval_id, session=session, principal=principal)
    except ApprovalError as exc:
        _audit_refusal(store, audit, approval_id, session, principal, exc.reason)
        raise
    action = _action_of(before) if before is not None else _action_of(approval)
    audit.record("denied", action, session=session, principal=principal, approval_id=approval_id)
    return approval


def _audit_refusal(
    store: ApprovalStore,
    audit: AuditLog,
    approval_id: str,
    session: str,
    principal: str,
    reason: str,
) -> None:
    approval = store.get(approval_id)
    action = (
        _action_of(approval)
        if approval is not None
        else Action(source="unknown", tool="unknown", args={})
    )
    audit.record(
        "blocked",
        action,
        session=session,
        principal=principal,
        approval_id=approval_id,
        reason=reason,
    )


def _error_kind(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ToolCallError):
        return "malformed_result" if "malformed" in str(exc) else "tool_error"
    if isinstance(exc, PermissionDenied):
        return "permission_denied"
    if isinstance(exc, KeyError | TypeError | ValidationError):
        return "unexpected_result"
    return "backend_error"


def _result_summary(action: Action, raw: Any) -> dict[str, Any]:
    if action.tool == "update_issue":
        return {"key": raw["key"], "status": raw["status"]}
    if action.tool == "create_draft":
        return {"draft_id": raw["draft_id"]}
    if action.tool == "send":
        return {"message_id": raw["message_id"]}
    if action.tool in {"put_memory", "mark_stale"}:
        return {"found": raw is not None}
    return {}


class ActionRunner:
    """Runs approved actions. MCP writes go through `call_json` with the grant; memory writes
    go to the memory backend directly (it is local state, but still a write)."""

    def __init__(
        self,
        *,
        store: ApprovalStore,
        audit: AuditLog,
        policy: Policy,
        data_mode: DataMode,
        clients: Mapping[Source, Client] | None = None,
        memory: MemoryBackend | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.store = store
        self.audit = audit
        self.policy = policy
        self.data_mode = data_mode
        self.clients: dict[str, Client] = {str(k): v for k, v in (clients or {}).items()}
        self.memory = memory
        self.timeout_s = timeout_s

    async def execute(
        self,
        approval_id: str,
        *,
        session: str,
        principal: str,
        action: Action | None = None,
    ) -> ActionOutcome:
        """Run the approved action (by default, the one recorded with the approval)."""
        approval = self.store.get(approval_id)
        if action is None:
            action = approval.action() if approval is not None else None
        if action is None:
            reason = "no such approval" if approval is None else _closed_reason(approval)
            return self._blocked(approval, approval_id, session, principal, reason)

        decision = self.policy.evaluate(action, self.data_mode)
        if decision.verdict != "require_approval":
            reason = f"policy: {decision.reason}"
            return self._blocked(approval, approval_id, session, principal, reason, action)
        try:
            grant = self.store.consume(approval_id, action, session=session, principal=principal)
        except ApprovalError as exc:
            return self._blocked(approval, approval_id, session, principal, exc.reason, action)

        self.audit.record(
            "attempted", action, session=session, principal=principal, approval_id=approval_id
        )
        try:
            with anyio.fail_after(self.timeout_s):
                raw = await self._dispatch(action, grant)
            result = _result_summary(action, raw)
        except Exception as exc:
            kind = _error_kind(exc)
            self.audit.record(
                "failed",
                action,
                session=session,
                principal=principal,
                approval_id=approval_id,
                reason=kind,
            )
            return ActionOutcome(approval_id=approval_id, status="failed", reason=kind)
        self.audit.record(
            "succeeded",
            action,
            session=session,
            principal=principal,
            approval_id=approval_id,
            result=result,
        )
        return ActionOutcome(approval_id=approval_id, status="succeeded", result=result)

    async def _dispatch(self, action: Action, grant: Grant) -> Any:
        if action.source == "memory":
            if self.memory is None:
                raise PermissionDenied("no memory backend configured")
            if action.tool == "put_memory":
                return self.memory.put(MemoryItem.model_validate(action.args))
            if action.tool == "mark_stale":
                return self.memory.mark_stale(
                    str(action.args["key"]),
                    str(action.args["reason"]),
                    action.args.get("superseded_by"),
                )
            raise PermissionDenied(f"unsupported memory tool {action.tool!r}")
        client = self.clients.get(action.source)
        if client is None:
            raise ToolCallError(f"no MCP connection for {action.source}")
        return await call_json(client, action.tool, action.args, self.timeout_s, grant=grant)

    def _blocked(
        self,
        approval: Approval | None,
        approval_id: str,
        session: str,
        principal: str,
        reason: str,
        action: Action | None = None,
    ) -> ActionOutcome:
        subject = action or (
            _action_of(approval)
            if approval is not None
            else Action(source="unknown", tool="unknown", args={})
        )
        self.audit.record(
            "blocked",
            subject,
            session=session,
            principal=principal,
            approval_id=approval_id,
            reason=reason,
        )
        return ActionOutcome(approval_id=approval_id, status="blocked", reason=reason)


def _closed_reason(approval: Approval) -> str:
    return {
        "denied": "approval was denied",
        "consumed": "approval already used",
        "expired": "approval expired",
    }.get(approval.status, "approval has no action to run")
