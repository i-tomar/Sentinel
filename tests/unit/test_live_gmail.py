"""Live API clients against mock transports (no network, no real accounts)."""

from __future__ import annotations

import base64
import calendar
import hashlib
from datetime import UTC, date, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from sentinel.live.gmail import (
    API as GMAIL_API,
)
from sentinel.live.gmail import (
    READONLY_SCOPE,
    TOKEN_URL,
    GmailAuth,
    GmailClient,
    build_query,
    check_scopes,
    consent_url,
    exchange_code,
    pkce_pair,
)
from sentinel.live.http import LiveAPIError
from sentinel.retrieval.normalize import gmail_document
from tests.live_payloads import (
    THREAD_ID,
    Recorder,
    gmail_thread,
)

pytestmark = [pytest.mark.unit, pytest.mark.anyio]

NOW = datetime(2026, 11, 2, tzinfo=UTC)


# --- Gmail --------------------------------------------------------------------------------


def test_scope_check_allows_only_readonly() -> None:
    check_scopes(READONLY_SCOPE)
    with pytest.raises(LiveAPIError, match=r"beyond gmail\.readonly"):
        check_scopes(f"{READONLY_SCOPE} https://www.googleapis.com/auth/gmail.send")
    with pytest.raises(LiveAPIError, match=r"beyond gmail\.readonly"):
        check_scopes(f"{READONLY_SCOPE} https://mail.google.com/")
    with pytest.raises(LiveAPIError, match="does not grant"):
        check_scopes("https://www.googleapis.com/auth/gmail.metadata")


def test_consent_url_requests_only_readonly_with_pkce() -> None:
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    url = urlparse(consent_url("cid.apps.test", "http://127.0.0.1:8765", challenge, "st"))
    params = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert params["scope"] == READONLY_SCOPE
    assert (params["code_challenge_method"], params["access_type"]) == ("S256", "offline")
    assert params["redirect_uri"] == "http://127.0.0.1:8765"
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    assert params["code_challenge"] == expected.decode().rstrip("=")


def _token_route(scope: str) -> Recorder:
    def route(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        form = parse_qs(request.content.decode())
        if form["grant_type"] == ["authorization_code"]:
            return httpx.Response(
                200,
                json={
                    "access_token": "at",
                    "refresh_token": "rt-123",
                    "scope": scope,
                    "expires_in": 3600,
                },
            )
        return httpx.Response(
            200, json={"access_token": "at-fresh", "scope": scope, "expires_in": 3600}
        )

    return Recorder(route)


async def test_code_exchange_refuses_broad_scopes() -> None:
    ok = _token_route(READONLY_SCOPE)
    assert (
        await exchange_code("cid", "sec", "code", "ver", "http://127.0.0.1:1", ok.transport)
        == "rt-123"
    )
    form = parse_qs(ok.requests[0].content.decode())
    assert form["code_verifier"] == ["ver"]
    broad = _token_route(f"{READONLY_SCOPE} https://www.googleapis.com/auth/gmail.modify")
    with pytest.raises(LiveAPIError, match=r"beyond gmail\.readonly"):
        await exchange_code("cid", "sec", "code", "ver", "http://127.0.0.1:1", broad.transport)


async def test_refresh_token_flow_is_cached_and_scope_checked() -> None:
    rec = _token_route(READONLY_SCOPE)
    auth = GmailAuth("cid", "sec", "rt", transport=rec.transport)
    assert await auth.access_token() == "at-fresh"
    assert await auth.access_token() == "at-fresh"
    assert len(rec.requests) == 1
    bad = GmailAuth(
        "cid", "sec", "rt", transport=_token_route("https://mail.google.com/").transport
    )
    with pytest.raises(LiveAPIError):
        await bad.access_token()


def test_gmail_query_uses_epoch_bounds_and_label() -> None:
    q = build_query("PROJ-147", date(2026, 10, 26), date(2026, 11, 2), "escalation")
    after = calendar.timegm((2026, 10, 26, 0, 0, 0))
    before = calendar.timegm((2026, 11, 2, 0, 0, 0))
    assert q == f"PROJ-147 after:{after} before:{before} label:escalation"


def _gmail_route(request: httpx.Request) -> httpx.Response:
    url = urlparse(str(request.url))
    assert request.headers["Authorization"] == "Bearer at-fresh"
    if url.path.endswith("/threads"):
        return httpx.Response(
            200,
            json={
                "threads": [{"id": THREAD_ID, "snippet": "Login fails &amp; more"}],
                "resultSizeEstimate": 1,
            },
        )
    if url.path.endswith(f"/threads/{THREAD_ID}"):
        assert parse_qs(url.query)["format"] == ["full"]
        return httpx.Response(200, json=gmail_thread())
    return httpx.Response(404, json={})


async def test_gmail_search_and_thread_mapping() -> None:
    auth = GmailAuth("cid", "sec", "rt", transport=_token_route(READONLY_SCOPE).transport)
    rec = Recorder(_gmail_route)
    client = GmailClient(auth, transport=rec.transport)
    hits, total = await client.search_threads("PROJ-147", None, None, None, limit=5, offset=0)
    assert ([h["id"] for h in hits], total, hits[0]["snippet"]) == (
        [THREAD_ID],
        1,
        "Login fails & more",
    )
    raw = await client.get_thread(THREAD_ID)
    doc = gmail_document(raw, NOW)
    assert doc.title == "SSO login failing"
    assert doc.author == "dana@globex.test"
    first, second = raw["messages"]
    assert first["to"] == ["sofia@acme.test", "ops@acme.test"]
    assert first["body"] == "Login fails since Monday.\nPROJ-147?"  # text/plain preferred
    assert second["body"] == "Escalated to  Kabir  & team."  # html fallback, tags stripped
    assert second["cc"] == ["kabir@acme.test"]
    assert raw["labels"] == ["inbox", "label_7", "sent"]
    assert {r.url.path for r in rec.requests} <= {
        f"{urlparse(GMAIL_API).path}/threads",
        f"{urlparse(GMAIL_API).path}/threads/{THREAD_ID}",
    }
    with pytest.raises(LiveAPIError, match="invalid Gmail thread id"):
        await client.get_thread("../labels")
