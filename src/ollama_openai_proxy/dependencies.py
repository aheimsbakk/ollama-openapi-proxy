"""Upstream client dependency for FastAPI dependency injection."""

from __future__ import annotations

from ollama_openai_proxy.client import UpstreamClient

# Global client instance, created at startup and closed at shutdown.
_upstream_client: UpstreamClient | None = None


def get_upstream_client() -> UpstreamClient:
    """Return the singleton upstream client, or raise if not initialized."""
    if _upstream_client is None:
        raise RuntimeError(
            "HTTP client not initialized. Call server.create_app() before handling requests."
        )
    return _upstream_client


def set_upstream_client(client: UpstreamClient) -> None:
    """Set the singleton upstream client (used during app creation)."""
    global _upstream_client
    _upstream_client = client
