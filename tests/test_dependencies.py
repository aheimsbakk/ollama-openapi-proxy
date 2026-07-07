"""Tests for the dependencies module."""

from __future__ import annotations

import pytest

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.dependencies import (
    get_upstream_client,
    set_upstream_client,
)


class TestGetUpstreamClient:
    """Tests for get_upstream_client."""

    def test_raises_when_not_initialized(self) -> None:
        """Calling get_upstream_client before initialization raises RuntimeError."""
        # Reset the global to simulate uninitialized state
        import ollama_openai_proxy.dependencies as deps

        original = deps._upstream_client
        try:
            deps._upstream_client = None
            with pytest.raises(RuntimeError) as exc_info:
                get_upstream_client()
            assert "UpstreamClient not initialized" in str(exc_info.value)
        finally:
            deps._upstream_client = original

    def test_returns_client_when_initialized(self) -> None:
        """Calling get_upstream_client after initialization returns the client."""
        import ollama_openai_proxy.dependencies as deps

        original = deps._upstream_client
        try:
            client = UpstreamClient(base_url="http://test", timeout_seconds=30)
            set_upstream_client(client)
            result = get_upstream_client()
            assert result is client
        finally:
            deps._upstream_client = original


class TestSetUpstreamClient:
    """Tests for set_upstream_client."""

    def test_sets_singleton(self) -> None:
        """set_upstream_client stores the client as the singleton."""
        import ollama_openai_proxy.dependencies as deps

        original = deps._upstream_client
        try:
            client = UpstreamClient(base_url="http://test", timeout_seconds=60)
            set_upstream_client(client)
            assert deps._upstream_client is client
        finally:
            deps._upstream_client = original
