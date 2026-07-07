"""Tests for the 4 streaming gaps identified in the blueprint audit.

Each test reproduces a specific defect between the streaming implementation
and the BLUEPRINT.md specification.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ollama_openai_proxy.translators import streaming


class _MockSSEResponse:
    """A minimal mock httpx.Response with aiter_bytes for streaming tests."""

    def __init__(self, sse_lines: list[str]) -> None:
        self._sse_lines = sse_lines
        self._closed = False

    async def aiter_bytes(self) -> Any:
        for line in self._sse_lines:
            yield (line + "\n").encode("utf-8")

    async def aclose(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Gap 1: Final streaming chunk — response field should be empty
# ---------------------------------------------------------------------------


class TestGap1FinalChunkEmptyResponse:
    """BLUEPRINT §A.2: final done:true object has response='' (Ollama convention)."""

    @pytest.mark.asyncio
    async def test_completions_final_chunk_response_is_empty(self) -> None:
        """Final done:true chunk for completions must have response='', not accumulated text."""
        sse_lines = [
            'data: {"choices": [{"text": "Hello"}]}',
            'data: {"choices": [{"text": " world"}]}',
            'data: {"choices": [{"text": "!", "finish_reason": "stop"}]}',
            'data: {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=False
        ):
            lines.append(line)

        final = json.loads(lines[-1].strip())
        assert final["done"] is True
        assert final["response"] == "", (
            f"Final response should be empty string, got: {final['response']!r}"
        )

    def test_build_chunk_final_completions_response_empty(self) -> None:
        """build_chunk(done=True) for completions must yield empty response."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        adapter.content = "accumulated text"
        adapter.finish_reason = "stop"
        adapter.usage = {"prompt_tokens": 5, "completion_tokens": 4}

        chunk = adapter.build_chunk(done=True)
        assert chunk["response"] == "", (
            f"build_chunk final response should be empty, got: {chunk['response']!r}"
        )

    def test_build_chunk_intermediate_completions_response_holds_content(self) -> None:
        """Intermediate (done=False) chunks must still include accumulated content."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        adapter.content = "partial"
        adapter.finish_reason = None

        chunk = adapter.build_chunk(done=False)
        assert chunk["response"] == "partial"


# ---------------------------------------------------------------------------
# Gap 2: created_at is empty in streaming because handlers pass created=None
# ---------------------------------------------------------------------------


class TestGap2CreatedAtNotEmpty:
    """BLUEPRINT §A.2: every streaming line has created_at from upstream timestamp."""

    def test_created_at_populated_from_timestamp(self) -> None:
        """Adapter must convert the Unix timestamp to ISO 8601 for created_at."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        assert adapter.created_at != "", (
            "created_at should be populated from created=1720000000"
        )
        assert "2024-07-03" in adapter.created_at

    def test_created_at_empty_when_none(self) -> None:
        """When created is None, created_at must be empty string."""
        adapter = streaming._SSEAdapter("test-model", None, is_chat=True)
        assert adapter.created_at == ""

    @pytest.mark.asyncio
    async def test_all_streaming_lines_have_created_at(self) -> None:
        """Every NDJSON line in a stream must include a non-empty created_at."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}',
            'data: {"usage": {"prompt_tokens": 2, "completion_tokens": 2}}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        for i, line in enumerate(lines):
            obj = json.loads(line.strip())
            assert obj.get("created_at", ""), (
                f"Line {i} has empty or missing created_at: {obj}"
            )


# ---------------------------------------------------------------------------
# Gap 3: Tool call arguments not parsed in streaming
# ---------------------------------------------------------------------------


class TestGap3ToolCallArgumentsParsed:
    """BLUEPRINT §A.3: OpenAI sends arguments as JSON string; Ollama expects parsed object."""

    def test_streaming_tool_call_arguments_parsed(self) -> None:
        """Streaming tool calls must have arguments as a parsed dict, not a string."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
            }
        ]
        adapter.finish_reason = "tool_calls"
        adapter.usage = {"prompt_tokens": 10, "completion_tokens": 5}

        chunk = adapter.build_chunk(done=True)
        args = chunk["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict), (
            f"Tool call arguments must be a dict, got {type(args).__name__}: {args!r}"
        )
        assert args == {"city": "Tokyo"}

    def test_streaming_tool_call_arguments_already_parsed(self) -> None:
        """If arguments are already a dict (edge case), they pass through unchanged."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": {"message": "hi"}},
            }
        ]
        adapter.finish_reason = "tool_calls"

        chunk = adapter.build_chunk(done=True)
        args = chunk["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"message": "hi"}

    def test_streaming_tool_call_arguments_empty_string(self) -> None:
        """Empty string arguments become an empty dict."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "noop", "arguments": ""},
            }
        ]
        adapter.finish_reason = "stop"

        chunk = adapter.build_chunk(done=True)
        args = chunk["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {}

    def test_streaming_tool_call_arguments_invalid_json(self) -> None:
        """Invalid JSON arguments fall back to empty dict."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "broken", "arguments": "{not json"},
            }
        ]
        adapter.finish_reason = "stop"

        chunk = adapter.build_chunk(done=True)
        args = chunk["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {}


# ---------------------------------------------------------------------------
# Gap 4: Accumulated content not emitted before error line on stream error
# ---------------------------------------------------------------------------


class TestGap4ContentEmittedBeforeError:
    """BLUEPRINT §7.2: on connection break, emit last content then error line."""

    @pytest.mark.asyncio
    async def test_stream_error_emits_content_before_error(self) -> None:
        """When upstream connection breaks, accumulated content is emitted first."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: {"choices": [{"delta": {"content": " again"}}]}',
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        # Should have: content lines + error line
        assert len(lines) >= 2, (
            f"Expected at least 2 lines (content + error), got {len(lines)}: {lines}"
        )

        # Last content line (before the error) should have accumulated content.
        # Find the last non-error line.
        last_content = None
        for line in lines:
            obj = json.loads(line.strip())
            if obj.get("done") is False:
                last_content = obj

        assert last_content is not None, "Expected at least one non-done content line"
        assert (
            last_content.get("message", {}).get("content", "") == "Hello world again"
        ), f"Last content line should have accumulated content, got: {last_content}"

        # Last line should be an error
        error_line = json.loads(lines[-1].strip())
        assert error_line["done"] is True
        assert "error" in error_line

    @pytest.mark.asyncio
    async def test_stream_error_emits_content_before_error_completions(self) -> None:
        """Same behavior for completions mode."""
        sse_lines = [
            'data: {"choices": [{"text": "The sky"}]}',
            'data: {"choices": [{"text": " is blue"}]}',
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=False
        ):
            lines.append(line)

        assert len(lines) >= 2
        # Last content line should have accumulated content.
        last_content = None
        for line in lines:
            obj = json.loads(line.strip())
            if obj.get("done") is False:
                last_content = obj
        assert last_content is not None
        assert last_content.get("response", "") == "The sky is blue", (
            f"Last content line should have accumulated content, got: {last_content}"
        )

        error_line = json.loads(lines[-1].strip())
        assert error_line["done"] is True
        assert "error" in error_line

    @pytest.mark.asyncio
    async def test_stream_error_no_content_emits_error_only(self) -> None:
        """If no content was accumulated, only the error line is emitted."""
        mock_response = _MockSSEResponse([])

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        # Should have exactly 1 error line (no content to emit)
        assert len(lines) == 1
        error_line = json.loads(lines[0].strip())
        assert error_line["done"] is True
        assert "error" in error_line
