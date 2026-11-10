"""Turn explicit write requests in the user's question into proposed actions.

Only the user's question is read here: never retrieved records, memory notes or model output,
so data cannot ask for a write. A proposal only opens an approval request (see
`sentinel.permissions.gate`); nothing runs until the user approves it with
`sentinel approve`, and the policy can still refuse it outright.

Recognised requests (deterministic, fixture demo of the flow):
- "set|mark|move PROJ-142 to|as Done"                 -> jira.update_issue (write)
- "add a comment to PROJ-142: <text>"                 -> jira.update_issue (write)
- "draft an email|reply to <address> about <topic>"   -> gmail.create_draft (write)
- "send draft-0001"                                   -> gmail.send (destructive)
"""

from __future__ import annotations

import re

from sentinel.permissions.actions import Action

_STATUSES = {
    "to do": "To Do",
    "in progress": "In Progress",
    "blocked": "Blocked",
    "in review": "In Review",
    "done": "Done",
}
_KEY = r"(?P<key>[A-Za-z][A-Za-z0-9]{1,9}-\d{1,6})"
_SET_STATUS = re.compile(
    rf"\b(?:set|mark|move)\s+{_KEY}\s+(?:to|as)\s+(?P<status>{'|'.join(_STATUSES)})\b", re.I
)
_COMMENT = re.compile(
    rf"\b(?:add\s+(?:a\s+)?comment\s+(?:to|on)|comment\s+on)\s+{_KEY}\s*[:,-]\s*(?P<text>\S.*)$",
    re.I | re.S,
)
_DRAFT = re.compile(
    r"\bdraft\s+(?:an?\s+)?(?:email|e-mail|reply|message)\s+to\s+"
    r"(?P<to>[\w.%+-]+@[\w-]+(?:\.[\w-]+)+)(?:\s+about\s+(?P<about>.+?))?\s*[.?!]?\s*$",
    re.I | re.S,
)
_SEND = re.compile(r"\bsend\s+(?P<draft>draft-\d{4})\b", re.I)


def actions_from_question(question: str) -> list[Action]:
    actions: list[Action] = []
    for s in _SET_STATUS.finditer(question):
        status = _STATUSES[s.group("status").lower()]
        actions.append(
            Action(
                source="jira",
                tool="update_issue",
                args={"key": s.group("key").upper(), "status": status},
            )
        )
    if c := _COMMENT.search(question):
        actions.append(
            Action(
                source="jira",
                tool="update_issue",
                args={"key": c.group("key").upper(), "comment": c.group("text").strip()},
            )
        )
    if d := _DRAFT.search(question):
        about = (d.group("about") or "").strip()
        actions.append(
            Action(
                source="gmail",
                tool="create_draft",
                args={
                    "to": [d.group("to").lower()],
                    "subject": (about or "Follow-up")[:200],
                    "body": f"Hi,\n\n{about or 'Following up.'}\n\nBest regards",
                },
            )
        )
    for m in _SEND.finditer(question):
        actions.append(
            Action(source="gmail", tool="send", args={"draft_id": m.group("draft").lower()})
        )
    unique: dict[str, Action] = {a.fingerprint(): a for a in actions}
    return list(unique.values())
