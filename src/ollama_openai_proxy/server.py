"""FastAPI application creation and startup/shutdown logic."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.config import Config
from ollama_openai_proxy.dependencies import get_upstream_client, set_upstream_client
from ollama_openai_proxy.errors import global_exception_handler
from ollama_openai_proxy.router import register_routes

logger = logging.getLogger("ollama_openai_proxy")


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application.

    Registers all routes, middleware, and lifecycle hooks.
    """
    app = FastAPI(
        title="Ollama-to-OpenAI Proxy",
        description="Drop-in Ollama API proxy that translates to/from an OpenAI-compatible upstream server.",
        version="0.0.0-proxy",
    )

    # CORS: allow all origins for drop-in compatibility with Ollama clients.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler — catches everything and returns Ollama error format.
    app.add_exception_handler(Exception, global_exception_handler)

    # Create the upstream HTTP client and register it as a dependency.
    upstream = UpstreamClient(timeout_seconds=config.request_timeout)
    set_upstream_client(upstream)

    # Register all Ollama API routes.
    register_routes(app, upstream)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        """Close the upstream client on shutdown."""
        if upstream:
            await upstream.close()

    return app
