"""Tests for the UpstreamClient HTTP client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.errors import AppError


class TestClientGet:
    """Tests for UpstreamClient.get()."""

    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        """A successful GET returns the JSON body."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            200,
            json={"models": [{"id": "test"}]},
            headers={"content-type": "application/json"},
        )
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.get("http://upstream/v1/models")

        assert result == {"models": [{"id": "test"}]}
        await client.close()

    @pytest.mark.asyncio
    async def test_get_connect_error(self) -> None:
        """A connection error raises AppError(502)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.get = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(AppError) as exc_info:
            await client.get("http://upstream/v1/models")

        assert exc_info.value.status_code == 502
        await client.close()

    @pytest.mark.asyncio
    async def test_get_timeout(self) -> None:
        """A timeout raises AppError(504)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.get = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

        with pytest.raises(AppError) as exc_info:
            await client.get("http://upstream/v1/models")

        assert exc_info.value.status_code == 504
        await client.close()

    @pytest.mark.asyncio
    async def test_get_non_200_json_error(self) -> None:
        """A non-200 response with JSON body raises AppError with the error message."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            500,
            json={"error": "upstream crashed"},
            headers={"content-type": "application/json"},
        )
        client._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.get("http://upstream/v1/models")

        assert exc_info.value.status_code == 500
        assert "upstream crashed" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_get_non_200_non_json_error(self) -> None:
        """A non-200 response with non-JSON body raises AppError with raw text."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            502,
            content=b"Bad Gateway",
            headers={"content-type": "text/plain"},
        )
        client._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.get("http://upstream/v1/models")

        assert exc_info.value.status_code == 502
        assert "Bad Gateway" in str(exc_info.value)
        await client.close()


class TestClientPost:
    """Tests for UpstreamClient.post()."""

    @pytest.mark.asyncio
    async def test_post_success(self) -> None:
        """A successful POST returns the JSON body."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            200,
            json={"model": "test", "response": "ok"},
            headers={"content-type": "application/json"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.post(
            "http://upstream/v1/completions", json={"prompt": "hi"}
        )

        assert result == {"model": "test", "response": "ok"}
        await client.close()

    @pytest.mark.asyncio
    async def test_post_connect_error(self) -> None:
        """A connection error raises AppError(502)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(AppError) as exc_info:
            await client.post("http://upstream/v1/completions", json={"prompt": "hi"})

        assert exc_info.value.status_code == 502
        await client.close()

    @pytest.mark.asyncio
    async def test_post_timeout(self) -> None:
        """A timeout raises AppError(504)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

        with pytest.raises(AppError) as exc_info:
            await client.post("http://upstream/v1/completions", json={"prompt": "hi"})

        assert exc_info.value.status_code == 504
        await client.close()

    @pytest.mark.asyncio
    async def test_post_non_200_json_error(self) -> None:
        """A non-200 response with JSON body raises AppError."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            429,
            json={"error": "rate limit exceeded"},
            headers={"content-type": "application/json"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.post("http://upstream/v1/completions", json={"prompt": "hi"})

        assert exc_info.value.status_code == 429
        assert "rate limit exceeded" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_post_non_200_non_json_error(self) -> None:
        """A non-200 response with non-JSON body falls back to raw text."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            503,
            content=b"<html>Service Unavailable</html>",
            headers={"content-type": "text/html"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.post("http://upstream/v1/completions", json={"prompt": "hi"})

        assert exc_info.value.status_code == 503
        assert "Service Unavailable" in str(exc_info.value)
        await client.close()


class TestClientStreamPost:
    """Tests for UpstreamClient.stream_post()."""

    @pytest.mark.asyncio
    async def test_stream_post_success(self) -> None:
        """A successful stream POST returns the raw httpx.Response."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            200,
            json={"dummy": True},
            headers={"content-type": "application/json"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        response = await client.stream_post(
            "http://upstream/v1/chat/completions", json={"stream": True}
        )

        assert response.status_code == 200
        assert response.json() == {"dummy": True}
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_post_connect_error(self) -> None:
        """A connection error raises AppError(502)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(AppError) as exc_info:
            await client.stream_post(
                "http://upstream/v1/chat/completions", json={"stream": True}
            )

        assert exc_info.value.status_code == 502
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_post_timeout(self) -> None:
        """A timeout raises AppError(504)."""
        client = UpstreamClient(timeout_seconds=30)
        client._client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

        with pytest.raises(AppError) as exc_info:
            await client.stream_post(
                "http://upstream/v1/chat/completions", json={"stream": True}
            )

        assert exc_info.value.status_code == 504
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_post_non_200_json_error(self) -> None:
        """A non-200 response with JSON body raises AppError with the error message."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            400,
            json={"error": "invalid model"},
            headers={"content-type": "application/json"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.stream_post(
                "http://upstream/v1/chat/completions", json={"stream": True}
            )

        assert exc_info.value.status_code == 400
        assert "invalid model" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_post_non_200_non_json_error(self) -> None:
        """A non-200 response with non-JSON body raises AppError with string representation."""
        client = UpstreamClient(timeout_seconds=30)
        mock_response = httpx.Response(
            500,
            content=b"Internal Server Error",
            headers={"content-type": "text/plain"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(AppError) as exc_info:
            await client.stream_post(
                "http://upstream/v1/chat/completions", json={"stream": True}
            )

        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in str(exc_info.value)
        await client.close()


class TestClientClose:
    """Tests for UpstreamClient.close()."""

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """close() calls aclose on the underlying client."""
        client = UpstreamClient(timeout_seconds=30)
        aclose_called = False

        async def mock_aclose() -> None:
            nonlocal aclose_called
            aclose_called = True

        client._client.aclose = mock_aclose
        await client.close()
        assert aclose_called is True
