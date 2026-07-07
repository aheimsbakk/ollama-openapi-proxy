"""Tests for the /api/chat endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import (
    SAMPLE_CHAT_REQUEST,
    SAMPLE_CHAT_RESPONSE,
    _MockUpstreamClient,
)


class TestChatNonStreaming:
    """Non-streaming /api/chat tests."""

    def test_chat_success(
        self, client: TestClient, mock_client: _MockUpstreamClient
    ) -> None:
        """A valid chat request returns an Ollama-format response with tool calls."""
        mock_client.on("POST", "/v1/chat/completions", SAMPLE_CHAT_RESPONSE)

        response = client.post("/api/chat", json=SAMPLE_CHAT_REQUEST)

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "llama-3.2-3b"
        assert body["done"] is True
        assert body["done_reason"] == "tool_calls"
        assert body["message"]["role"] == "assistant"
        assert "tool_calls" in body["message"]
        assert body["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert body["message"]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo",
        }
        assert body["prompt_eval_count"] == 45
        assert body["eval_count"] == 15

    def test_chat_missing_model(self, client: TestClient) -> None:
        """Missing model field returns 400."""
        response = client.post("/api/chat", json={"messages": []})
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_chat_missing_messages(self, client: TestClient) -> None:
        """Missing messages field returns 400."""
        response = client.post("/api/chat", json={"model": "test"})
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_chat_invalid_json(self, client: TestClient) -> None:
        """Invalid JSON body returns 400."""
        response = client.post(
            "/api/chat",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "detail" in response.json()
