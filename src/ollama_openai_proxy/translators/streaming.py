"""SSE-to-NDJSON stream adapter.

Reads SSE events from an upstream OpenAI response and yields Ollama-format
NDJSON lines. Implements the state machine described in BLUEPRINT.md Appendix B.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger("ollama_openai_proxy")


async def sse_to_ollama_stream(
    upstream_response: httpx.Response,
    model: str,
    created: int | float | None,
    is_chat: bool,
) -> AsyncIterator[str]:
    """Convert an upstream SSE stream to Ollama NDJSON lines.

    Args:
        upstream_response: The raw httpx.Response from the upstream server.
        model: The model name to echo in each Ollama output line.
        created: The Unix timestamp from the upstream response.
        is_chat: If True, use chat completion format; otherwise use completions format.

    Yields:
        Ollama-format NDJSON lines, one per SSE event.
    """
    adapter = _SSEAdapter(model, created, is_chat)

    try:
        async for line in _read_sse_lines(upstream_response):
            action = adapter.process_line(line)

            if action == "emit":
                chunk = adapter.build_chunk(done=False)
                yield json.dumps(chunk, separators=(",", ":")) + "\n"

            elif action == "done":
                final = adapter.build_chunk(done=True)
                yield json.dumps(final, separators=(",", ":")) + "\n"

            elif action == "error":
                error_obj = {
                    "model": model,
                    "error": adapter.last_error or "upstream error in stream",
                    "done": True,
                }
                yield json.dumps(error_obj, separators=(",", ":")) + "\n"

            elif action == "skip":
                logger.warning("Skipping malformed SSE line: %s", line[:100])

    except httpx.ReadTimeout:
        last_chunk = adapter.build_chunk(done=True)
        yield json.dumps(last_chunk, separators=(",", ":")) + "\n"
        error_obj = {
            "model": model,
            "error": "upstream request timed out",
            "done": True,
        }
        yield json.dumps(error_obj, separators=(",", ":")) + "\n"

    except httpx.StreamError as exc:
        if adapter.content or adapter.tool_calls:
            last_chunk = adapter.build_chunk(done=False)
            yield json.dumps(last_chunk, separators=(",", ":")) + "\n"
        error_obj = {
            "model": model,
            "error": "upstream connection lost mid-response",
            "done": True,
        }
        yield json.dumps(error_obj, separators=(",", ":")) + "\n"

    finally:
        await upstream_response.aclose()


class _SSEAdapter:
    """Stateful adapter for converting OpenAI SSE chunks to Ollama NDJSON.

    Maintains internal accumulators for content, tool calls, usage, and
    finish reason across SSE events.
    """

    def __init__(self, model: str, created: int | float | None, is_chat: bool) -> None:
        self.model = model
        self.created_at = _ts_to_iso8601(created) if created else ""
        self.is_chat = is_chat
        self.content: str = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.usage: dict[str, Any] = {}
        self.finish_reason: str | None = None
        self.last_error: str | None = None

    def process_line(self, line: str) -> str:
        """Process a single SSE line.

        Returns one of: "emit", "done", "error", "skip".
        """
        # Empty line = SSE event boundary
        if not line or line.isspace():
            return "skip"

        # SSE data line
        if line.startswith("data: "):
            data = line[6:]

            # [DONE] marker
            if data.strip() == "[DONE]":
                return "done"

            # Try to parse as JSON
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                return "skip"

            # Check for error in the chunk
            if isinstance(parsed, dict) and "error" in parsed:
                self.last_error = parsed["error"]
                return "error"

            # Accumulate state from the chunk
            self._accumulate(parsed)
            return "emit"

        # SSE comment or metadata field
        if (
            line.startswith(":")
            or line.startswith("id:")
            or line.startswith("event:")
            or line.startswith("retry:")
        ):
            return "skip"

        return "skip"

    def _accumulate(self, chunk: dict[str, Any]) -> None:
        """Accumulate state from an OpenAI streaming chunk."""
        # Usage may arrive in a chunk without choices (some upstreams send it separately)
        chunk_usage = chunk.get("usage")
        if chunk_usage:
            self.usage.update(chunk_usage)

        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})

        # Content accumulation
        if "content" in delta:
            delta_content = delta.get("content")
            if delta_content:
                self.content += delta_content
        elif "text" in choice:
            delta_text = choice.get("text")
            if delta_text:
                self.content += delta_text

        # Tool call deltas
        delta_tool_calls = delta.get("tool_calls", choice.get("tool_calls", []))
        if delta_tool_calls:
            for tc_delta in delta_tool_calls:
                idx = tc_delta.get("index", 0)
                while len(self.tool_calls) <= idx:
                    self.tool_calls.append({})
                self._merge_tool_call(self.tool_calls[idx], tc_delta)

        # Finish reason
        fr = choice.get("finish_reason")
        if fr:
            self.finish_reason = fr

    def _merge_tool_call(self, existing: dict[str, Any], delta: dict[str, Any]) -> None:
        """Merge a tool call delta into an existing tool call entry."""
        if "function" in delta:
            if "function" not in existing:
                existing["function"] = {}
            func_delta = delta["function"]
            if "name" in func_delta and func_delta["name"]:
                existing["function"]["name"] = func_delta["name"]
            if "arguments" in func_delta:
                existing["function"]["arguments"] = (
                    existing["function"].get("arguments", "") + func_delta["arguments"]
                )
        if "id" in delta and delta["id"]:
            existing["id"] = delta["id"]
        if "type" in delta and delta["type"]:
            existing["type"] = delta["type"]

    def build_chunk(self, done: bool = False) -> dict[str, Any]:
        """Build an Ollama-format response object."""
        if self.is_chat:
            message: dict[str, Any] = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                message["tool_calls"] = self.tool_calls
            result: dict[str, Any] = {
                "model": self.model,
                "created_at": self.created_at,
                "message": message,
                "done": done,
            }
        else:
            result = {
                "model": self.model,
                "created_at": self.created_at,
                "response": self.content,
                "done": done,
            }

        if done:
            result["done_reason"] = self.finish_reason or "stop"
            result["total_duration"] = 0
            result["load_duration"] = 0
            result["prompt_eval_count"] = self.usage.get("prompt_tokens", 0)
            result["prompt_eval_duration"] = 0
            result["eval_count"] = self.usage.get("completion_tokens", 0)
            result["eval_duration"] = 0

        return result


async def _read_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Read lines from an SSE response, handling \\r\\n and \\n line endings."""
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            yield line
    if buffer.strip():
        yield buffer.rstrip("\r")


def _ts_to_iso8601(unix_ts: int | float | None) -> str:
    """Convert a Unix timestamp to an ISO 8601 string."""
    if unix_ts is None:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(unix_ts)))
    except (OSError, ValueError, OverflowError):
        return ""
