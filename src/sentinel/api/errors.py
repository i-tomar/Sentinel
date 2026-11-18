"""Consistent API error envelope. Never include stack traces, secrets, or env."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sentinel.api.models import ErrorBody, ErrorEnvelope


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.headers = headers or {}
        super().__init__(message)


def request_id_of(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else "unknown"


def error_response(
    *,
    code: str,
    message: str,
    request_id: str,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message, request_id=request_id))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers or None,
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(
        code=exc.code,
        message=exc.message,
        request_id=request_id_of(request),
        status_code=exc.status_code,
        headers=exc.headers,
    )


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    message = detail if isinstance(detail, str) else "request failed"
    code = "not_found" if exc.status_code == 404 else "http_error"
    return error_response(
        code=code,
        message=message,
        request_id=request_id_of(request),
        status_code=exc.status_code,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Surface a short message only; never dump raw body or env.
    errs = exc.errors()
    first: dict[str, Any] = errs[0] if errs else {}
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = str(first.get("msg", "invalid request"))
    message = f"{loc}: {msg}" if loc else msg
    return error_response(
        code="validation_error",
        message=message,
        request_id=request_id_of(request),
        status_code=422,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    del exc  # never leak exception text or traceback to clients
    return error_response(
        code="internal_error",
        message="internal error",
        request_id=request_id_of(request),
        status_code=500,
    )
