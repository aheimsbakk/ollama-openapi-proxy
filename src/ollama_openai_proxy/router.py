"""Route registration for all Ollama API endpoints.

Maps each Ollama API path to its handler function. Unsupported endpoints
return HTTP 501. Unknown paths return HTTP 404.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.errors import AppError
from ollama_openai_proxy.handlers import chat, embed, generate, models

logger = logging.getLogger("ollama_openai_proxy")


def register_routes(app: FastAPI, client: UpstreamClient) -> None:
    """Register all Ollama API routes on the given FastAPI application.

    Each route calls its handler with the FastAPI Request object and the
    shared UpstreamClient instance.
    """
    # --- Primary inference endpoints ---
    app.add_api_route(
        "/api/generate",
        generate.handle_generate,
        methods=["POST"],
        tags=["inference"],
        response_model=None,
    )
    logger.debug("Route registered: POST /api/generate")

    app.add_api_route(
        "/api/chat",
        chat.handle_chat,
        methods=["POST"],
        tags=["inference"],
        response_model=None,
    )
    logger.debug("Route registered: POST /api/chat")

    # --- Embedding endpoints ---
    app.add_api_route(
        "/api/embed",
        embed.handle_embed,
        methods=["POST"],
        tags=["embedding"],
        response_model=None,
    )
    logger.debug("Route registered: POST /api/embed")

    app.add_api_route(
        "/api/embeddings",
        embed.handle_embeddings_legacy,
        methods=["POST"],
        tags=["embedding"],
        response_model=None,
    )
    logger.debug("Route registered: POST /api/embeddings (legacy)")

    # --- Informational endpoints ---
    app.add_api_route(
        "/api/tags", models.handle_tags, methods=["GET"], tags=["informational"]
    )
    logger.debug("Route registered: GET /api/tags")
    app.add_api_route(
        "/api/show", models.handle_show, methods=["POST"], tags=["informational"]
    )
    logger.debug("Route registered: POST /api/show")
    app.add_api_route(
        "/api/ps", models.handle_ps, methods=["GET"], tags=["informational"]
    )
    logger.debug("Route registered: GET /api/ps")
    app.add_api_route(
        "/api/version", models.handle_version, methods=["GET"], tags=["informational"]
    )
    logger.debug("Route registered: GET /api/version")

    # --- Unsupported endpoints (return 501) ---
    unsupported = [
        ("POST", "/api/create"),
        ("POST", "/api/copy"),
        ("DELETE", "/api/delete"),
        ("POST", "/api/pull"),
        ("POST", "/api/push"),
        ("HEAD", "/api/blobs/{digest}"),
        ("POST", "/api/blobs/{digest}"),
    ]
    for method, path in unsupported:
        app.add_api_route(
            path,
            _unsupported_handler,
            methods=[method],
            tags=["unsupported"],
        )
        logger.debug("Route registered (unsupported): %s %s", method, path)

    # --- Catch-all for unknown paths ---
    # FastAPI doesn't have a native catch-all, so we use a fallback route.
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def _catch_all(request: Request, path: str) -> JSONResponse:
        """Return 404 for any path not matched by the routes above."""
        full_path = f"/{path}"
        logger.warning("Unknown path requested: %s %s", request.method, full_path)
        return JSONResponse(
            status_code=404,
            content={"error": "not found"},
        )


async def _unsupported_handler(
    request: Request, digest: str | None = None
) -> JSONResponse:
    """Handler for unsupported Ollama API endpoints.

    Returns HTTP 501 with a descriptive error message.
    """
    path = request.url.path
    logger.warning("Unsupported endpoint called: %s %s", request.method, path)
    return JSONResponse(
        status_code=501,
        content={"error": f"Endpoint not supported: {path}"},
    )
