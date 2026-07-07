"""Tests for the /api/generate endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import (
    SAMPLE_GENERATE_REQUEST,
    SAMPLE_GENERATE_RESPONSE,
    _MockUpstreamClient,
)


class TestGenerateNonStreaming:
    """Non-streaming /api/generate tests."""

    def test_generate_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid generate request returns an Ollama-format response."""
        mock_client.on("POST", "/v1/completions", SAMPLE_GENERATE_RESPONSE)

        response = client.post("/api/generate", json=SAMPLE_GENERATE_REQUEST)

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "llama-3.2-3b"
        assert (
            body["response"] == "The sky appears blue because of Rayleigh scattering."
        )
        assert body["done"] is True
        assert body["done_reason"] == "stop"
        assert body["prompt_eval_count"] == 8
        assert body["eval_count"] == 43

    def test_generate_missing_model(self, client: TestClient) -> None:
        """Missing model field returns 400."""
        response = client.post("/api/generate", json={"prompt": "hello"})
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_generate_without_prompt(self, client: TestClient) -> None:
        """Request without prompt (model load/unload) is valid — prompt is optional."""
        response = client.post("/api/generate", json={"model": "llama-3.2-3b"})
        # No 400 — prompt is optional; request is forwarded to upstream
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "llama-3.2-3b"

    def test_generate_invalid_json(self, client: TestClient) -> None:
        """Invalid JSON body returns 400."""
        response = client.post(
            "/api/generate",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "detail" in response.json()
