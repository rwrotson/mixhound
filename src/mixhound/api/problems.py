from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

PROBLEM_MEDIA_TYPE = "application/problem+json"


class Problem(BaseModel):
    """RFC 9457 problem details returned for every error response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: Any = None
    instance: str | None = None
    errors: list[dict[str, Any]] | None = None


def problem_response(
    status: int,
    *,
    detail: object = None,
    instance: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build a problem details response."""
    problem = Problem(
        title=HTTPStatus(status).phrase,
        status=status,
        detail=detail,
        instance=instance,
        errors=errors,
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=headers,
        media_type=PROBLEM_MEDIA_TYPE,
    )


async def _http_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, HTTPException):
        raise TypeError("Expected HTTPException")
    return problem_response(
        exc.status_code, detail=exc.detail, instance=request.url.path, headers=exc.headers
    )


async def _validation_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):
        raise TypeError("Expected RequestValidationError")
    return problem_response(
        422,
        detail="Request validation failed",
        instance=request.url.path,
        errors=jsonable_encoder(exc.errors()),
    )


def register_error_handlers(application: FastAPI) -> None:
    """Render HTTP and validation errors as problem details."""
    application.add_exception_handler(HTTPException, _http_error)
    application.add_exception_handler(RequestValidationError, _validation_error)
