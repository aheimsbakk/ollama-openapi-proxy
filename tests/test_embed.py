"""Tests for the /api/embed and /api/embeddings endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import (
    SAMPLE_EMBED_REQUEST,
    SAMPLE_EMBED_RESPONSE,
    _MockUpstreamClient,
)


class TestEmbed:
    """POST /api/embed tests."""

    def test_embed_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid embed request returns an Ollama-format response."""
        mock_client.on("POST", "/v1/embeddings", SAMPLE_EMBED_RESPONSE)

        response = client.post("/api/embed", json=SAMPLE_EMBED_REQUEST)

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "all-minilm"
        assert len(body["embeddings"]) == 2
        assert body["embeddings"][0] == [
            0.01,
            -0.002,
            0.05,
            0.047,
            0.055,
            0.009,
            0.105,
            -0.026,
        ]
        assert body["prompt_eval_count"] == 12

    def test_embed_missing_model(self, client: TestClient) -> None:
        """Missing model field returns 400."""
        response = client.post("/api/embed", json={"input": "hello"})
        assert response.status_code == 400
        assert "detail" in response.json()


class TestEmbeddingsLegacy:
    """POST /api/embeddings (legacy) tests."""

    def test_embeddings_legacy_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid legacy embeddings request returns a flat embedding list."""
        legacy_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "all-minilm",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        mock_client.on("POST", "/v1/embeddings", legacy_response)

        response = client.post(
            "/api/embeddings", json={"model": "all-minilm", "prompt": "hello"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "all-minilm"
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert "embeddings" not in body  # legacy returns flat list

    def test_embeddings_legacy_missing_model(self, client: TestClient) -> None:
        """Missing model field returns 400."""
        response = client.post("/api/embeddings", json={"prompt": "hello"})
        assert response.status_code == 400
        assert "detail" in response.json()
