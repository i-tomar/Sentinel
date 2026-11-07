"""Risk -> decision. Pure function of the trusted registry, the action and configuration.

- unknown or unclassified tool: deny;
- read: allow;
- write: require approval; destructive: require approval plus a typed confirmation;
- invalid arguments, recipients/assignees outside ALLOWED_EMAIL_DOMAINS, or writes to a live
  source: deny without asking (live connectors are read-only by design).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from sentinel.mcp_clients.registry import Risk, lookup
from sentinel.permissions.actions import Action

Verdict = Literal["allow", "require_approval", "deny"]
DataMode = Literal["fixture", "live"]


class Decision(BaseModel):
    verdict: Verdict
    reason: str
    risk: Risk | None = None
    needs_confirmation: bool = False
    confirmation: str | None = None
    """What the user must type to approve a destructive action."""


def _outside(addresses: Iterable[str], allowed: tuple[str, ...]) -> int:
    return sum(1 for a in addresses if a.rsplit("@", 1)[-1].lower() not in allowed)


@dataclass(frozen=True)
class Policy:
    allowed_email_domains: tuple[str, ...] = ("acme.test",)

    def evaluate(self, action: Action, data_mode: DataMode) -> Decision:
        spec = lookup(action.source, action.tool)
        if spec is None:
            return Decision(verdict="deny", reason="unknown or unclassified tool")
        if spec.risk == "read":
            return Decision(verdict="allow", reason="read-only tool", risk="read")
        deny = Decision(verdict="deny", reason="", risk=spec.risk)
        if spec.args is not None:
            try:
                spec.args.model_validate(action.args)
            except ValidationError as exc:
                fields = sorted({".".join(map(str, e["loc"])) or "args" for e in exc.errors()})
                return deny.model_copy(update={"reason": f"invalid arguments: {', '.join(fields)}"})
        if data_mode == "live" and spec.source != "memory":
            return deny.model_copy(update={"reason": "writes to live sources are not supported"})
        recipients = [str(a) for a in action.args.get("to") or []]
        assignee = action.args.get("assignee")
        if isinstance(assignee, str) and "@" in assignee:
            recipients.append(assignee)
        outside = _outside(recipients, self.allowed_email_domains)
        if outside:
            return deny.model_copy(
                update={"reason": f"{outside} address(es) outside ALLOWED_EMAIL_DOMAINS"}
            )
        if spec.risk == "destructive":
            target = next(iter(action.args.values()), "")
            return Decision(
                verdict="require_approval",
                reason="destructive tool: approval and typed confirmation required",
                risk="destructive",
                needs_confirmation=True,
                confirmation=f"{action.tool} {target}".strip(),
            )
        return Decision(verdict="require_approval", reason="write tool", risk="write")
