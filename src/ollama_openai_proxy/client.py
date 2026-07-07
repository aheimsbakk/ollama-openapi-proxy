"""Async HTTP client for upstream OpenAI-compatible server calls."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ollama_openai_proxy.errors import AppError

logger = logging.getLogger("ollama_openai_proxy")


def _truncate_body(body: dict[str, Any] | None, max_len: int = 500) -> str:
    """Truncate a JSON body to max_len characters for logging."""
    if body is None:
        return "None"
    text = str(body)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


class UpstreamClient:
    """Async HTTP client with timeout and error handling for upstream calls.

    Creates a single ``httpx.AsyncClient`` instance for connection pooling.
    All calls use the configured timeout.
    """

    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(
            connect=10, read=timeout_seconds, write=10, pool=10
        )
        self._client = httpx.AsyncClient(timeout=self._timeout)

    @property
    def base_url(self) -> str:
        """Return the upstream base URL."""
        return self._base_url

    async def get(self, url: str) -> dict[str, Any]:
        """Send a GET request and return the JSON body as a dict."""
        logger.info("Upstream GET %s", url)
        try:
            response = await self._client.get(url)
        except httpx.ConnectError as exc:
            logger.error("Upstream GET %s — connection failed: %s", url, exc)
            raise AppError(f"AI server unreachable: {exc}", status_code=502) from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "Upstream GET %s — timed out after %ss", url, self._timeout.read
            )
            raise AppError(
                f"Request to AI server timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        logger.debug("Upstream GET %s — status=%s", url, response.status_code)

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
            logger.warning(
                "Upstream GET %s — non-200: %s — %s",
                url,
                response.status_code,
                error_msg,
            )
            raise AppError(
                f"AI server returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        return response.json()

    async def post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a POST request with a JSON body and return the response body as a dict."""
        logger.info("Upstream POST %s", url)
        logger.debug("Upstream POST %s — body: %s", url, _truncate_body(json))

        try:
            response = await self._client.post(url, json=json)
        except httpx.ConnectError as exc:
            logger.error("Upstream POST %s — connection failed: %s", url, exc)
            raise AppError(f"AI server unreachable: {exc}", status_code=502) from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "Upstream POST %s — timed out after %ss", url, self._timeout.read
            )
            raise AppError(
                f"Request to AI server timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        logger.debug("Upstream POST %s — status=%s", url, response.status_code)

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
            logger.warning(
                "Upstream POST %s — non-200: %s — %s",
                url,
                response.status_code,
                error_msg,
            )
            raise AppError(
                f"AI server returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        return response.json()

    async def stream_post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Send a POST request and return the raw httpx.Response for streaming.

        The caller is responsible for reading and closing the response stream.
        """
        logger.info("Upstream streaming POST %s", url)
        logger.debug("Upstream streaming POST %s — body: %s", url, _truncate_body(json))

        try:
            response = await self._client.post(url, json=json, follow_redirects=False)
        except httpx.ConnectError as exc:
            logger.error("Upstream streaming POST %s — connection failed: %s", url, exc)
            raise AppError(f"AI server unreachable: {exc}", status_code=502) from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "Upstream streaming POST %s — timed out after %ss",
                url,
                self._timeout.read,
            )
            raise AppError(
                f"Request to AI server timed out after {self._timeout.read} seconds",
                status_code=504,
            ) from exc

        logger.debug(
            "Upstream streaming POST %s — status=%s", url, response.status_code
        )

        if response.status_code != 200:
            body_text = await response.aread()
            try:
                body = response.json()
            except Exception:
                body = body_text.decode(errors="replace")
            error_msg = (
                body.get("error", str(body)) if isinstance(body, dict) else str(body)
            )
            logger.warning(
                "Upstream streaming POST %s — non-200: %s — %s",
                url,
                response.status_code,
                error_msg,
            )
            raise AppError(
                f"AI server returned {response.status_code}: {error_msg}",
                status_code=response.status_code,
            )

        logger.debug("Upstream streaming POST %s — stream opened", url)
        return response

    async def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        logger.debug("Closing upstream HTTP client")
        await self._client.aclose()
