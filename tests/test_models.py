"""Tests for the model informational endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ollama_openai_proxy import __version__
from tests.conftest import (
    SAMPLE_MODELS_RESPONSE,
    _MockUpstreamClient,
)


class TestTags:
    """GET /api/tags tests."""

    def test_tags_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid tags request returns Ollama-format model list."""
        mock_client.on("GET", "/v1/models", SAMPLE_MODELS_RESPONSE)

        response = client.get("/api/tags")

        assert response.status_code == 200
        body = response.json()
        assert "models" in body
        assert len(body["models"]) == 2
        assert body["models"][0]["name"] == "llama-3.2-3b"
        assert body["models"][0]["details"]["format"] == "gguf"
        assert body["models"][0]["details"]["family"] == "llama"
        assert body["models"][1]["details"]["family"] == "mistral"

    def test_tags_empty(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """An empty model list returns an empty models array."""
        mock_client.on("GET", "/v1/models", {"object": "list", "data": []})

        response = client.get("/api/tags")

        assert response.status_code == 200
        body = response.json()
        assert body["models"] == []


class TestShow:
    """POST /api/show tests."""

    def test_show_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid show request returns Ollama-format model details."""
        mock_client.on(
            "GET",
            "/v1/models",
            {
                "object": "list",
                "data": [
                    {
                        "id": "llama-3.2-3b",
                        "object": "model",
                        "created": 1720000000,
                        "owned_by": "library",
                    },
                ],
            },
        )

        response = client.post("/api/show", json={"model": "llama-3.2-3b"})

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "llama-3.2-3b"
        assert body["details"]["format"] == "gguf"

    def test_show_missing_model(self, client: TestClient) -> None:
        """Missing model field returns 400."""
        response = client.post("/api/show", json={})
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_show_model_not_found(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """Requesting a non-existent model returns 404."""
        mock_client.on("GET", "/v1/models", {"object": "list", "data": []})

        response = client.post("/api/show", json={"model": "nonexistent-model"})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestPs:
    """GET /api/ps tests."""

    def test_ps_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid ps request returns models with expires_at and size_vram."""
        mock_client.on("GET", "/v1/models", SAMPLE_MODELS_RESPONSE)

        response = client.get("/api/ps")

        assert response.status_code == 200
        body = response.json()
        assert "models" in body
        assert body["models"][0]["expires_at"] == "2099-12-31T23:59:59Z"
        assert body["models"][0]["size_vram"] == 0


class TestVersion:
    """GET /api/version tests."""

    def test_version(self, client: TestClient) -> None:
        """The version endpoint returns a static response."""
        response = client.get("/api/version")

        assert response.status_code == 200
        body = response.json()
        assert body["version"] == __version__
