"""Error response builders for the proxy."""

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ollama_openai_proxy")


class AppError(Exception):
    """Application-level error with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def build_error_response(exc: Exception) -> JSONResponse:
    """Convert an exception into an Ollama-compatible JSON error response.

    Returns a JSON body of the form ``{"error": "<message>"}``.
    """
    if isinstance(exc, AppError):
        status_code = exc.status_code
        message = exc.message
        logger.warning(
            "AppError — status=%s message=%s",
            status_code,
            message,
        )
    elif isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        logger.warning(
            "HTTPException — status=%s message=%s",
            status_code,
            message,
        )
    else:
        logger.exception("Unhandled exception during request processing")
        status_code = 500
        message = "internal server error"

    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any exception that escapes a request handler.

    Returns an Ollama-compatible error response so a single failing request
    does not crash the server.
    """
    return build_error_response(exc)
