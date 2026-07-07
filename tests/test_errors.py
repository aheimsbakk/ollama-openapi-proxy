"""Tests for error handling and boundary conditions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ollama_openai_proxy.errors import AppError, build_error_response


class TestAppError:
    """Tests for the AppError exception class."""

    def test_default_status_code(self) -> None:
        exc = AppError("something failed")
        assert exc.status_code == 500
        assert exc.message == "something failed"

    def test_custom_status_code(self) -> None:
        exc = AppError("not found", status_code=404)
        assert exc.status_code == 404

    def test_string_representation(self) -> None:
        exc = AppError("test error")
        assert str(exc) == "test error"


class TestBuildErrorResponse:
    """Tests for the build_error_response function."""

    def test_app_error(self) -> None:
        exc = AppError("upstream unreachable", status_code=502)
        response = build_error_response(exc)
        assert response.status_code == 502
        assert response.body == b'{"error":"upstream unreachable"}'

    def test_unknown_exception(self) -> None:
        exc = RuntimeError("unexpected error")
        response = build_error_response(exc)
        assert response.status_code == 500
        body = response.body.decode()
        assert '"error"' in body


class TestUnsupportedEndpoints:
    """Tests for unsupported endpoint handling."""

    def test_api_create_returns_501(self, client: TestClient) -> None:
        response = client.post("/api/create", json={"model": "test"})
        assert response.status_code == 501
        assert "error" in response.json()

    def test_api_copy_returns_501(self, client: TestClient) -> None:
        response = client.post("/api/copy", json={"source": "a", "destination": "b"})
        assert response.status_code == 501
        assert "error" in response.json()

    def test_api_delete_returns_501(self, client: TestClient) -> None:
        response = client.request(
            "DELETE",
            "/api/delete",
            content='{"model": "test"}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 501
        assert "error" in response.json()

    def test_api_pull_returns_501(self, client: TestClient) -> None:
        response = client.post("/api/pull", json={"model": "test"})
        assert response.status_code == 501
        assert "error" in response.json()

    def test_api_push_returns_501(self, client: TestClient) -> None:
        response = client.post("/api/push", json={"model": "test"})
        assert response.status_code == 501
        assert "error" in response.json()

    def test_api_blobs_post_returns_501(self, client: TestClient) -> None:
        response = client.post("/api/blobs/sha256:abc123")
        assert response.status_code == 501
        assert "error" in response.json()

    def test_unknown_path_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/unknown")
        assert response.status_code == 404
        assert response.json()["error"] == "not found"
