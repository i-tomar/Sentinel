"""One-time Gmail consent for installed apps: loopback redirect + PKCE, gmail.readonly only.

You run this yourself (`sentinel live gmail-auth`); it opens Google's consent page, receives
the redirect on 127.0.0.1, exchanges the code and returns the refresh token.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import anyio
import httpx

from sentinel.live.gmail import consent_url, exchange_code, pkce_pair
from sentinel.live.http import LiveAPIError

_PAGE = b"<html><body>Sentinel received the Google consent. You can close this tab.</body></html>"


class LoopbackReceiver:
    """Listens on 127.0.0.1:<free port> for exactly one OAuth redirect."""

    def __init__(self) -> None:
        self.params: dict[str, str] = {}
        self.received = threading.Event()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                query = parse_qs(urlparse(self.path).query)
                receiver.params = {k: v[0] for k, v in query.items()}
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE)
                receiver.received.set()

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.redirect_uri = f"http://127.0.0.1:{self.server.server_port}"

    def wait(self, timeout_s: float) -> dict[str, str]:
        thread = threading.Thread(target=self.server.handle_request, daemon=True)
        self.server.timeout = timeout_s
        thread.start()
        if not self.received.wait(timeout_s):
            self.server.server_close()
            raise LiveAPIError("timed out waiting for the Google consent redirect")
        thread.join(5)
        self.server.server_close()
        return self.params


async def obtain_refresh_token(
    client_id: str,
    client_secret: str,
    open_browser: Callable[[str], object],
    timeout_s: float = 300.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    receiver = LoopbackReceiver()
    open_browser(consent_url(client_id, receiver.redirect_uri, challenge, state))
    params = await anyio.to_thread.run_sync(receiver.wait, timeout_s)
    if params.get("state") != state:
        raise LiveAPIError("consent redirect had the wrong state; aborting")
    if "error" in params:
        raise LiveAPIError(f"consent refused: {params['error']}")
    code = params.get("code")
    if not code:
        raise LiveAPIError("consent redirect carried no code")
    return await exchange_code(
        client_id, client_secret, code, verifier, receiver.redirect_uri, transport
    )
