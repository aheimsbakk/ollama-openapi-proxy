"""Configuration loaded from CLI arguments and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable proxy configuration.

    Resolution order (highest precedence first):
    1. CLI arguments
    2. Environment variables
    3. Defaults
    """

    listen_host: str = "0.0.0.0"
    listen_port: int = 11434
    upstream_url: str = "http://localhost:8080/v1"
    request_timeout: int = 300
    verbosity: int = 0  # 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG


def load_config(
    *,
    host: str | None = None,
    port: int | None = None,
    upstream_url: str | None = None,
    timeout: int | None = None,
    verbosity: int = 0,
) -> Config:
    """Build a Config instance, merging CLI args with env vars and defaults.

    CLI arguments override environment variables, which override defaults.
    """
    listen_host = host or os.environ.get("LISTEN_HOST", "0.0.0.0")
    listen_port = int(port or os.environ.get("LISTEN_PORT", "11434"))
    upstream = upstream_url or os.environ.get(
        "UPSTREAM_URL", "http://localhost:8080/v1"
    )
    req_timeout = int(timeout or os.environ.get("REQUEST_TIMEOUT", "300"))

    return Config(
        listen_host=listen_host,
        listen_port=listen_port,
        upstream_url=upstream,
        request_timeout=req_timeout,
        verbosity=verbosity,
    )
