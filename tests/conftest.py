"""Shared test fixtures for the proxy test suite."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.config import Config, load_config
from ollama_openai_proxy.dependencies import get_upstream_client
from ollama_openai_proxy.server import create_app


# ---------------------------------------------------------------------------
# Sample payloads (from BLUEPRINT.md Appendix A)
# ---------------------------------------------------------------------------

SAMPLE_GENERATE_REQUEST: dict[str, Any] = {
    "model": "llama-3.2-3b",
    "prompt": "Why is the sky blue?",
    "stream": False,
    "options": {
        "temperature": 0.7,
        "seed": 42,
        "num_predict": 100,
    },
}

SAMPLE_GENERATE_RESPONSE: dict[str, Any] = {
    "id": "cmpl-abc123",
    "object": "text_completion",
    "created": 1720000000,
    "model": "llama-3.2-3b",
    "choices": [
        {
            "text": "The sky appears blue because of Rayleigh scattering.",
            "index": 0,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 43,
        "total_tokens": 51,
    },
}

SAMPLE_CHAT_REQUEST: dict[str, Any] = {
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "stream": False,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ],
}

SAMPLE_CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-xyz",
    "object": "chat.completion",
    "created": 1720000002,
    "model": "llama-3.2-3b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 15,
        "total_tokens": 60,
    },
}

SAMPLE_EMBED_REQUEST: dict[str, Any] = {
    "model": "all-minilm",
    "input": ["Why is the sky blue?", "Why is grass green?"],
    "truncate": True,
}

SAMPLE_EMBED_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.01, -0.002, 0.05, 0.047, 0.055, 0.009, 0.105, -0.026],
        },
        {
            "object": "embedding",
            "index": 1,
            "embedding": [-0.01, 0.06, 0.025, -0.006, 0.073, 0.017, 0.09, -0.052],
        },
    ],
    "model": "all-minilm",
    "usage": {
        "prompt_tokens": 12,
        "total_tokens": 12,
    },
}

SAMPLE_MODELS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "id": "llama-3.2-3b",
            "object": "model",
            "created": 1720000000,
            "owned_by": "library",
        },
        {
            "id": "mistral-7b",
            "object": "model",
            "created": 1719900000,
            "owned_by": "library",
        },
    ],
}


class _MockUpstreamClient(UpstreamClient):
    """A mock UpstreamClient that returns predefined responses."""

    def __init__(self) -> None:
        # Skip parent __init__ — we don't need a real httpx client
        self._timeout = None
        self._client = _MockHTTPClient()
        # Response overrides keyed by (method, path)
        self._responses: dict[tuple[str, str], dict[str, Any]] = {}

    def on(self, method: str, path: str, response: dict[str, Any]) -> None:
        """Register a response for a specific method/path combination."""
        self._responses[(method, path)] = response

    async def get(self, url: str) -> dict[str, Any]:
        return self._get_response("GET", url)

    async def post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._get_response("POST", url)

    async def stream_post(
        self, url: str, json: dict[str, Any] | None = None
    ) -> httpx.Response:
        raise NotImplementedError("stream_post not supported in tests")

    async def close(self) -> None:
        pass

    def _get_response(self, method: str, url: str) -> dict[str, Any]:
        for (m, path), response in self._responses.items():
            if url.endswith(path):
                return response
        raise RuntimeError(f"No mock response registered for {method} {url}")


class _MockHTTPClient:
    """A minimal mock HTTP client for base_url access."""

    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:19999/v1"


def _mock_upstream_client() -> _MockUpstreamClient:
    """Dependency override factory for tests."""
    client = _MockUpstreamClient()
    # Register default responses
    client.on("GET", "/v1/models", SAMPLE_MODELS_RESPONSE)
    client.on("POST", "/v1/completions", SAMPLE_GENERATE_RESPONSE)
    client.on("POST", "/v1/chat/completions", SAMPLE_CHAT_RESPONSE)
    client.on("POST", "/v1/embeddings", SAMPLE_EMBED_RESPONSE)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    """Return a Config with a mock upstream URL."""
    return load_config(
        host="127.0.0.1",
        port=11434,
        upstream_url="http://127.0.0.1:19999/v1",
        timeout=30,
        verbosity=0,
    )


@pytest.fixture
def mock_client() -> _MockUpstreamClient:
    """Return a mock upstream client with default responses."""
    return _mock_upstream_client()


@pytest.fixture
def app(config: Config, mock_client: _MockUpstreamClient):
    """Create a FastAPI app with the mock client dependency overridden."""
    app = create_app(config)
    app.dependency_overrides[get_upstream_client] = lambda: mock_client
    return app


@pytest.fixture
def client(app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)
