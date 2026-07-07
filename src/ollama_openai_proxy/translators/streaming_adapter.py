"""Stateful SSE-to-Ollama chunk adapter.

Maintains internal accumulators for content, tool calls, usage, and
finish reason across SSE events. Used by the streaming generator in
``streaming.py``.
"""

from __future__ import annotations

import json
import time
from typing import Any


def _ts_to_iso8601(unix_ts: int | float | None) -> str:
    """Convert a Unix timestamp to an ISO 8601 string.

    Uses second-precision with .000000Z to match Ollama's sub-second format.
    """
    if unix_ts is None:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime(int(unix_ts)))
    except (OSError, ValueError, OverflowError):
        return ""


class SSEAdapter:
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
        elif "reasoning_content" in delta:
            # llama.cpp sends reasoning_content instead of content
            delta_reasoning = delta.get("reasoning_content")
            if delta_reasoning:
                self.content += delta_reasoning
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

    def _parse_tool_arguments(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse arguments from JSON strings to dicts for each tool call."""
        result = []
        for tc in tool_calls:
            parsed = dict(tc)
            func = tc.get("function", {})
            if func:
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        parsed["function"] = {**func, "arguments": json.loads(args)}
                    except (json.JSONDecodeError, TypeError):
                        parsed["function"] = {**func, "arguments": {}}
                # If already a dict, leave as-is
                else:
                    parsed["function"] = func
            result.append(parsed)
        return result

    def build_chunk(self, done: bool = False) -> dict[str, Any]:
        """Build an Ollama-format response object."""
        if self.is_chat:
            message: dict[str, Any] = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                # Parse arguments from JSON strings to dicts (BLUEPRINT §A.3).
                message["tool_calls"] = self._parse_tool_arguments(self.tool_calls)
            result: dict[str, Any] = {
                "model": self.model,
                "created_at": self.created_at,
                "message": message,
                "done": done,
            }
        else:
            # Final chunk: response must be empty per Ollama convention
            # (client already has content from intermediate chunks).
            response_text = "" if done else self.content
            result = {
                "model": self.model,
                "created_at": self.created_at,
                "response": response_text,
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
