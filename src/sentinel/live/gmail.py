"""Read-only Gmail API client mapped to Sentinel's Gmail record shape.

Auth is OAuth 2.0 for installed apps: a refresh token obtained once with
`sentinel live gmail-auth` (loopback redirect + PKCE) is exchanged for short-lived access
tokens. Only the `gmail.readonly` scope is ever requested, and any token whose granted
scopes include anything else is refused. API requests are limited to GET users/me/threads.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
import secrets
import time
from datetime import UTC, date, datetime
from email.utils import getaddresses, parseaddr
from typing import Any
from urllib.parse import urlencode

import httpx

from sentinel.live.http import LiveAPIError, guarded_client, request_json

READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 - an endpoint, not a secret
API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_SEARCH_PAGES = 10
_THREAD_ID = re.compile(r"^[0-9a-f]{6,32}$")
_TAGS = re.compile(r"<[^>]+>")


def check_scopes(granted: str) -> None:
    scopes = set(granted.split())
    if READONLY_SCOPE not in scopes:
        raise LiveAPIError("Gmail token does not grant gmail.readonly")
    extra = sorted(scopes - {READONLY_SCOPE})
    if extra:
        raise LiveAPIError(f"refusing Gmail token with scopes beyond gmail.readonly: {extra}")


# --- OAuth (installed app, loopback + PKCE) -------------------------------------------------


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.decode().rstrip("=")


def consent_url(client_id: str, redirect_uri: str, challenge: str, state: str) -> str:
    return (
        AUTH_URL
        + "?"
        + urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": READONLY_SCOPE,
                "access_type": "offline",
                "prompt": "consent",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
            }
        )
    )


def _token_client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    return guarded_client((("POST", TOKEN_URL),), transport=transport)


async def exchange_code(
    client_id: str,
    client_secret: str,
    code: str,
    verifier: str,
    redirect_uri: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Trade the consent code for a refresh token, refusing anything beyond gmail.readonly."""
    async with _token_client(transport) as http:
        data = await request_json(
            http,
            "POST",
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    check_scopes(str(data.get("scope", "")))
    refresh = data.get("refresh_token")
    if not refresh:
        raise LiveAPIError("Google did not return a refresh token")
    return str(refresh)


class GmailAuth:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._form = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        self._transport = transport
        self._token: str | None = None
        self._expires = 0.0

    async def access_token(self) -> str:
        if self._token and time.monotonic() < self._expires - 60:
            return self._token
        async with _token_client(self._transport) as http:
            data = await request_json(http, "POST", TOKEN_URL, data=self._form)
        check_scopes(str(data.get("scope", "")))
        self._token = str(data["access_token"])
        self._expires = time.monotonic() + float(data.get("expires_in", 3600))
        return self._token


# --- mapping -----------------------------------------------------------------------------


def build_query(query: str, after: date | None, before: date | None, label: str | None) -> str:
    parts = [query.strip()] if query.strip() else []
    if after:
        parts.append(
            f"after:{int(datetime(after.year, after.month, after.day, tzinfo=UTC).timestamp())}"
        )
    if before:
        parts.append(
            f"before:{int(datetime(before.year, before.month, before.day, tzinfo=UTC).timestamp())}"
        )
    if label:
        parts.append(f"label:{label}")
    return " ".join(parts)


def _decode(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", "replace")


def message_text(payload: dict[str, Any]) -> str:
    """Prefer text/plain parts; fall back to tag-stripped text/html."""
    plain: list[str] = []
    rich: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime == "text/plain":
            plain.append(_decode(data))
        elif data and mime == "text/html":
            rich.append(html.unescape(_TAGS.sub(" ", _decode(data))))
        for child in part.get("parts") or []:
            walk(child)

    walk(payload)
    text = "\n".join(plain) if plain else "\n".join(rich)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in payload.get("headers") or []}


def thread_record(thread: dict[str, Any]) -> dict[str, Any]:
    messages = []
    labels: set[str] = set()
    subject = ""
    for msg in thread.get("messages") or []:
        payload = msg.get("payload") or {}
        headers = _headers(payload)
        subject = subject or headers.get("subject", "")
        labels |= {lbl.lower() for lbl in msg.get("labelIds") or []}
        sender = parseaddr(headers.get("from", ""))[1] or headers.get("from", "unknown")
        messages.append(
            {
                "sender": sender,
                "to": [a for _, a in getaddresses([headers.get("to", "")]) if a],
                "cc": [a for _, a in getaddresses([headers.get("cc", "")]) if a],
                "sent_at": datetime.fromtimestamp(int(msg["internalDate"]) / 1000, UTC).isoformat(),
                "body": message_text(payload),
            }
        )
    return {
        "id": thread["id"],
        "url": f"https://mail.google.com/mail/u/0/#all/{thread['id']}",
        "subject": subject or "(no subject)",
        "labels": sorted(labels),
        "messages": messages,
    }


class GmailClient:
    def __init__(self, auth: GmailAuth, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.auth = auth
        self.http = guarded_client((("GET", f"{API}/threads"),), transport=transport)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _get(self, url: str, params: dict[str, Any]) -> Any:
        token = await self.auth.access_token()
        return await request_json(
            self.http, "GET", url, params=params, headers={"Authorization": f"Bearer {token}"}
        )

    async def search_threads(
        self,
        query: str,
        after: date | None,
        before: date | None,
        label: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        q = build_query(query, after, before, label)
        wanted = offset + limit
        found: list[dict[str, Any]] = []
        token: str | None = None
        for _ in range(MAX_SEARCH_PAGES):
            params: dict[str, Any] = {"q": q, "maxResults": 100}
            if token:
                params["pageToken"] = token
            data = await self._get(f"{API}/threads", params)
            found += data.get("threads", [])
            token = data.get("nextPageToken")
            if not token or len(found) >= wanted:
                break
        hits = [
            {
                "id": t["id"],
                "url": f"https://mail.google.com/mail/u/0/#all/{t['id']}",
                "snippet": html.unescape(t.get("snippet", "")),
            }
            for t in found[offset:wanted]
        ]
        return hits, len(found) + (1 if token else 0)

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        if not _THREAD_ID.match(thread_id):
            raise LiveAPIError(f"invalid Gmail thread id: {thread_id!r}")
        thread = await self._get(f"{API}/threads/{thread_id}", {"format": "full"})
        return thread_record(thread)
