"""FastAPI app: fixture/mock ask + observed stage events for a future UI."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from sentinel.api.errors import (
    ApiError,
    api_error_handler,
    http_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from sentinel.api.models import MAX_QUESTION_CHARS, CreateRunRequest, CreateRunResponse
from sentinel.api.rate_limit import RateLimiter
from sentinel.api.runtime import DemoRuntime, events_payload, health_payload, result_payload

_RATE = RateLimiter(max_calls=30, window_s=60.0)
_OVERRIDE_HEADERS = frozenset({"x-llm-provider", "x-llm-model", "x-data-mode", "x-injection-guard"})


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response


def create_app(runtime: DemoRuntime | None = None) -> FastAPI:
    demo = runtime or DemoRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await demo.startup()
        try:
            yield
        finally:
            await demo.shutdown()

    app = FastAPI(
        title="Sentinel demo API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.runtime = demo
    app.add_middleware(RequestIdMiddleware)
    # Starlette types handlers as Exception; our callables take the concrete type.
    app.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return health_payload()

    @app.post("/api/runs", status_code=201)
    async def create_run(body: CreateRunRequest, request: Request) -> CreateRunResponse:
        client = request.client.host if request.client else "unknown"
        if not _RATE.allow(client):
            raise ApiError("rate_limited", "too many runs; try again later", status_code=429)
        # Reject oversized even if a client bypasses model max (defense in depth).
        if len(body.question) > MAX_QUESTION_CHARS:
            raise ApiError(
                "question_too_long",
                f"question must be at most {MAX_QUESTION_CHARS} characters",
                status_code=400,
            )
        # Demo safety: refuse explicit override headers (body already forbids extras).
        header_names = {h.lower() for h in request.headers}
        if header_names & _OVERRIDE_HEADERS:
            raise ApiError(
                "override_forbidden",
                "live provider, model, and data-mode overrides are not allowed",
                status_code=400,
            )
        try:
            run = await demo.run_ask(body.question, thread_id=body.thread_id)
        except Exception as exc:
            raise ApiError("run_failed", "ask failed", status_code=500) from exc
        return CreateRunResponse(run_id=run.run_id, status=run.status, mode="fixture")

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, object]:
        run = demo.store.get(run_id)
        if run is None:
            raise ApiError("not_found", "run not found", status_code=404)
        return result_payload(run)

    @app.get("/api/runs/{run_id}/events")
    async def get_events(run_id: str) -> dict[str, object]:
        run = demo.store.get(run_id)
        if run is None:
            raise ApiError("not_found", "run not found", status_code=404)
        return events_payload(run)

    return app


app = create_app()
