"""Tests for the SSE-to-NDJSON streaming adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from ollama_openai_proxy.translators import streaming


class TestSSEAdapter:
    """Tests for the _SSEAdapter state machine."""

    def test_process_data_line_emit(self) -> None:
        """A valid data line returns 'emit'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line('data: {"choices": [{"text": "hello"}]}')
        assert result == "emit"
        assert adapter.content == "hello"

    def test_process_data_line_done(self) -> None:
        """The [DONE] marker returns 'done'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("data: [DONE]")
        assert result == "done"

    def test_process_data_line_error(self) -> None:
        """A chunk with an error key returns 'error'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line('data: {"error": "something went wrong"}')
        assert result == "error"
        assert adapter.last_error == "something went wrong"

    def test_process_data_line_malformed_json(self) -> None:
        """Malformed JSON returns 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("data: {invalid json}")
        assert result == "skip"

    def test_process_empty_line(self) -> None:
        """An empty line returns 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("")
        assert result == "skip"

    def test_process_sse_comment(self) -> None:
        """An SSE comment returns 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line(": comment")
        assert result == "skip"

    def test_accumulate_chat_content(self) -> None:
        """Chat content deltas are accumulated."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"choices": [{"delta": {"content": "Hello"}}]})
        adapter._accumulate({"choices": [{"delta": {"content": " world"}}]})
        assert adapter.content == "Hello world"

    def test_accumulate_completions_content(self) -> None:
        """Completions text deltas are accumulated."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        adapter._accumulate({"choices": [{"text": "Hello"}]})
        adapter._accumulate({"choices": [{"text": " world"}]})
        assert adapter.content == "Hello world"

    def test_accumulate_tool_calls(self) -> None:
        """Tool call deltas are merged by index."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        adapter._accumulate(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"city"'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        adapter._accumulate(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ':"Tokyo"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        assert len(adapter.tool_calls) == 1
        assert adapter.tool_calls[0]["id"] == "call_abc"
        assert adapter.tool_calls[0]["function"]["name"] == "get_weather"
        assert adapter.tool_calls[0]["function"]["arguments"] == '{"city":"Tokyo"}'

    def test_accumulate_finish_reason(self) -> None:
        """Finish reason is captured from the final chunk."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"choices": [{"delta": {"content": "hi"}}]})
        adapter._accumulate({"choices": [{"finish_reason": "stop"}]})
        assert adapter.finish_reason == "stop"

    def test_accumulate_usage(self) -> None:
        """Usage stats are accumulated."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        assert adapter.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_build_chunk_non_streaming_chat(self) -> None:
        """Building a done chunk for chat includes usage stats."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.content = "Hello"
        adapter.finish_reason = "stop"
        adapter.usage = {"prompt_tokens": 8, "completion_tokens": 4}
        chunk = adapter.build_chunk(done=True)
        assert chunk["done"] is True
        assert chunk["done_reason"] == "stop"
        assert chunk["message"]["content"] == "Hello"
        assert chunk["prompt_eval_count"] == 8
        assert chunk["eval_count"] == 4

    def test_build_chunk_non_streaming_completions(self) -> None:
        """Building a done chunk for completions includes response field."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        adapter.content = "The sky is blue"
        adapter.finish_reason = "stop"
        adapter.usage = {"prompt_tokens": 5, "completion_tokens": 4}
        chunk = adapter.build_chunk(done=True)
        assert chunk["done"] is True
        assert chunk["response"] == "The sky is blue"
        assert "message" not in chunk


class TestReadSSELines:
    """Tests for the SSE line reader."""

    @pytest.mark.asyncio
    async def test_reads_lines(self) -> None:
        """The reader yields individual lines from byte chunks."""
        # Simulate a response with multiple SSE lines
        raw_bytes = b'data: {"choices": [{"text": "hello"}]}\n\ndata: [DONE]\n'

        # Create a mock response with aiter_bytes
        mock_response = httpx.Response(200, content=raw_bytes)
        mock_response._content = raw_bytes

        lines = []
        async for line in streaming._read_sse_lines(mock_response):
            lines.append(line)

        assert 'data: {"choices": [{"text": "hello"}]}' in lines
        assert "data: [DONE]" in lines


class TestTsToIso8601:
    """Tests for the Unix timestamp to ISO 8601 converter."""

    def test_valid_timestamp(self) -> None:
        result = streaming._ts_to_iso8601(1720000000)
        assert "2024-07-03" in result

    def test_none_timestamp(self) -> None:
        result = streaming._ts_to_iso8601(None)
        assert result == ""

    def test_zero_timestamp(self) -> None:
        result = streaming._ts_to_iso8601(0)
        assert "1970-01-01" in result
