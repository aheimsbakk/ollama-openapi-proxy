"""Additional streaming adapter edge-case tests."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from ollama_openai_proxy.translators import streaming


class TestProcessLineMetadataFields:
    """Tests for SSE metadata field handling in process_line."""

    def test_process_id_field(self) -> None:
        """Lines starting with 'id:' return 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("id: event-123")
        assert result == "skip"

    def test_process_event_field(self) -> None:
        """Lines starting with 'event:' return 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("event: chunk")
        assert result == "skip"

    def test_process_retry_field(self) -> None:
        """Lines starting with 'retry:' return 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("retry: 5000")
        assert result == "skip"

    def test_process_unknown_line(self) -> None:
        """An unrecognized line returns 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("random line content")
        assert result == "skip"


class TestTsToIso8601EdgeCases:
    """Tests for _ts_to_iso8601 error handling."""

    def test_float_timestamp(self) -> None:
        """A float timestamp is converted correctly."""
        result = streaming._ts_to_iso8601(1720000000.5)
        assert "2024-07-03" in result


class TestReadSSELinesBuffer:
    """Tests for leftover buffer handling in _read_sse_lines."""

    @pytest.mark.asyncio
    async def test_leftover_buffer_yields(self) -> None:
        """A non-newline-terminated final chunk is still yielded."""
        raw_bytes = b'data: {"choices": [{"text": "partial"}]'
        mock_response = httpx.Response(200, content=raw_bytes)
        mock_response._content = raw_bytes

        lines = []
        async for line in streaming._read_sse_lines(mock_response):
            lines.append(line)

        assert len(lines) == 1
        assert lines[0] == 'data: {"choices": [{"text": "partial"}]'


class TestSSEAdapterBuildChunkToolCalls:
    """Tests for build_chunk with accumulated tool calls."""

    def test_build_chunk_with_tool_calls(self) -> None:
        """A done chat chunk with tool calls includes them in the message."""
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
        assert chunk["done"] is True
        assert chunk["done_reason"] == "tool_calls"
        assert "tool_calls" in chunk["message"]
        assert chunk["message"]["tool_calls"][0]["id"] == "call_abc"
        assert chunk["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


class TestSSEAdapterAccumulateEmptyChoices:
    """Tests for _accumulate with edge-case chunks."""

    def test_accumulate_no_choices(self) -> None:
        """A chunk without choices is ignored (except usage)."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"usage": {"prompt_tokens": 5}})
        assert adapter.content == ""
        assert adapter.usage == {"prompt_tokens": 5}

    def test_accumulate_empty_choices_list(self) -> None:
        """An empty choices list is ignored."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"choices": []})
        assert adapter.content == ""

    def test_accumulate_delta_content_none(self) -> None:
        """None content delta does not append."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"choices": [{"delta": {"content": None}}]})
        assert adapter.content == ""

    def test_accumulate_text_none(self) -> None:
        """None text delta does not append (completions)."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        adapter._accumulate({"choices": [{"text": None}]})
        assert adapter.content == ""

    def test_accumulate_null_finish_reason(self) -> None:
        """A null finish_reason does not overwrite existing one."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.finish_reason = "stop"
        adapter._accumulate({"choices": [{"finish_reason": None}]})
        assert adapter.finish_reason == "stop"


class TestSSEAdapterMergeToolCall:
    """Tests for _merge_tool_call."""

    def test_merge_id_only(self) -> None:
        """Merging a delta with only an id sets it."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [{}]
        adapter._merge_tool_call(adapter.tool_calls[0], {"id": "call_1"})
        assert adapter.tool_calls[0]["id"] == "call_1"

    def test_merge_type_only(self) -> None:
        """Merging a delta with only a type sets it."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [{}]
        adapter._merge_tool_call(adapter.tool_calls[0], {"type": "function"})
        assert adapter.tool_calls[0]["type"] == "function"

    def test_merge_empty_function_name(self) -> None:
        """An empty function name is not merged."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter.tool_calls = [{"function": {"name": "existing"}}]
        adapter._merge_tool_call(adapter.tool_calls[0], {"function": {"name": ""}})
        assert adapter.tool_calls[0]["function"]["name"] == "existing"


class TestSSEAdapterProcessLineEdgeCases:
    """Tests for process_line edge cases."""

    def test_process_whitespace_only_line(self) -> None:
        """A whitespace-only line returns 'skip'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("   ")
        assert result == "skip"

    def test_process_data_with_only_space(self) -> None:
        """A data line with only whitespace after 'data: ' is not [DONE]."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("data:    ")
        assert result == "skip"

    def test_process_done_with_whitespace(self) -> None:
        """A [DONE] line with surrounding whitespace returns 'done'."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("data:  [DONE]  ")
        assert result == "done"

    def test_process_data_empty_json_object(self) -> None:
        """An empty JSON object is treated as a valid chunk."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=False)
        result = adapter.process_line("data: {}")
        assert result == "emit"
        assert adapter.content == ""


class TestSSEAdapterAccumulateToolCallsFromChoice:
    """Tests for tool_calls read from choice level (not delta)."""

    def test_tool_calls_on_choice_not_delta(self) -> None:
        """Tool calls at the choice level (not inside delta) are merged."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate(
            {
                "choices": [
                    {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]}
                ]
            }
        )
        assert len(adapter.tool_calls) == 1
        assert adapter.tool_calls[0]["id"] == "call_1"


class TestSSEAdapterAccumulateMultipleToolCalls:
    """Tests for accumulating multiple tool calls."""

    def test_multiple_tool_call_indices(self) -> None:
        """Tool calls with different indices are stored separately."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_0"},
                                {"index": 1, "id": "call_1"},
                            ]
                        }
                    }
                ]
            }
        )
        assert len(adapter.tool_calls) == 2
        assert adapter.tool_calls[0]["id"] == "call_0"
        assert adapter.tool_calls[1]["id"] == "call_1"


class TestSSEAdapterAccumulateUsageUpdate:
    """Tests for usage accumulation behavior."""

    def test_usage_replaced_on_new_chunk(self) -> None:
        """New usage replaces, not merges, the accumulator."""
        adapter = streaming._SSEAdapter("test-model", 1720000000, is_chat=True)
        adapter._accumulate({"usage": {"prompt_tokens": 5, "completion_tokens": 2}})
        adapter._accumulate({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        assert adapter.usage == {"prompt_tokens": 10, "completion_tokens": 5}


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


class TestSSEToOllamaStream:
    """Integration tests for the full sse_to_ollama_stream async generator."""

    @pytest.mark.asyncio
    async def test_stream_chat_completions(self) -> None:
        """A full chat streaming session yields NDJSON lines and calls aclose."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        assert len(lines) == 2
        first = lines[0].strip()
        chunk = streaming.json.loads(first)
        assert chunk["done"] is False
        assert chunk["message"]["content"] == "Hello"
        last = lines[1].strip()
        final = streaming.json.loads(last)
        assert final["done"] is True
        assert final["done_reason"] == "stop"
        assert mock_response._closed is True

    @pytest.mark.asyncio
    async def test_stream_completions_format(self) -> None:
        """A completions stream uses the response field instead of message."""
        sse_lines = [
            'data: {"choices": [{"text": "world"}]}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=False
        ):
            lines.append(line)

        assert len(lines) == 2
        chunk = streaming.json.loads(lines[0].strip())
        assert "response" in chunk
        assert chunk["response"] == "world"
        assert "message" not in chunk

    @pytest.mark.asyncio
    async def test_stream_error_in_chunk(self) -> None:
        """An error chunk in the stream yields an error NDJSON line."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "partial"}}]}',
            'data: {"error": "upstream crashed"}',
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        assert len(lines) == 2
        error_line = streaming.json.loads(lines[1].strip())
        assert error_line["error"] == "upstream crashed"
        assert error_line["done"] is True

    @pytest.mark.asyncio
    async def test_stream_skips_sse_comments(self) -> None:
        """SSE comments are skipped without producing output."""
        sse_lines = [
            ": this is a comment",
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_stream_uses_default_finish_reason(self) -> None:
        """When no finish_reason arrives, the final object uses 'stop'."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "text"}}]}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        final = streaming.json.loads(lines[-1].strip())
        assert final["done_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_accumulates_usage_in_final(self) -> None:
        """Usage stats from chunks appear in the final done:true object."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
            'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}',
            "data: [DONE]",
        ]
        mock_response = _MockSSEResponse(sse_lines)

        lines = []
        async for line in streaming.sse_to_ollama_stream(
            mock_response, model="test-model", created=1720000000, is_chat=True
        ):
            lines.append(line)

        final = streaming.json.loads(lines[-1].strip())
        assert final["prompt_eval_count"] == 5
        assert final["eval_count"] == 3
