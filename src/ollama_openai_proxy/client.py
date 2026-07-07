"""Async HTTP client for upstream OpenAI-compatible server calls."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ollama_openai_proxy.errors import AppError

logger = logging.getLogger("ollama_openai_proxy")


class UpstreamClient:
    """Async HTTP client with timeout and error handling for upstream calls.

    Creates a single ``httpx.AsyncClient`` instance for connection pooling.
    All calls use the configured timeout.
    """

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout = httpx.Timeout(
            connect=10, read=timeout_seconds, write=10, pool=10
        )
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def get(self, url: str) -> dict[str, Any]:
        """Send a GET request and return the JSON body as a dict."""
        try:
            response = await self._client.get(url)
        except httpx.ConnectError as exc:
            raise AppError(
                f"upstream server unreachable: {exc}", status_code=502
            ) from exc
        except httpx.TimeoutException as exc:
            raise AppError(
                f"upstream request timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type") == "application/json"
                else {}
            )
            error_msg = (
                body.get("error", response.text)
                if isinstance(body, dict)
                else response.text
            )
            raise AppError(
                f"upstream returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        return response.json()

    async def post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a POST request with a JSON body and return the response body as a dict."""
        try:
            response = await self._client.post(url, json=json)
        except httpx.ConnectError as exc:
            raise AppError(
                f"upstream server unreachable: {exc}", status_code=502
            ) from exc
        except httpx.TimeoutException as exc:
            raise AppError(
                f"upstream request timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type") == "application/json"
                else {}
            )
            error_msg = (
                body.get("error", response.text)
                if isinstance(body, dict)
                else response.text
            )
            raise AppError(
                f"upstream returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        return response.json()

    async def stream_post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Send a POST request and return the raw httpx.Response for streaming.

        The caller is responsible for reading and closing the response stream.
        """
        try:
            response = await self._client.post(url, json=json, follow_redirects=False)
        except httpx.ConnectError as exc:
            raise AppError(
                f"upstream server unreachable: {exc}", status_code=502
            ) from exc
        except httpx.TimeoutException as exc:
            raise AppError(
                f"upstream request timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        if response.status_code != 200:
            body_text = await response.aread()
            try:
                body = response.json()
            except Exception:
                body = body_text.decode(errors="replace")
            error_msg = (
                body.get("error", str(body))
                if isinstance(body, (dict, str))
                else str(body)
            )
            raise AppError(
                f"upstream returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        return response

    async def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        await self._client.aclose()
