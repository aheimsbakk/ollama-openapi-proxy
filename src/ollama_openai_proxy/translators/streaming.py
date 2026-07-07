"""SSE-to-NDJSON stream adapter.

Reads SSE events from an upstream OpenAI response and yields Ollama-format
NDJSON lines. Implements the state machine described in BLUEPRINT.md
Appendix B.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from ollama_openai_proxy.translators.streaming_adapter import SSEAdapter, _ts_to_iso8601

# Keep the old private name for backward compatibility with tests.
_SSEAdapter = SSEAdapter

logger = logging.getLogger("ollama_openai_proxy")


def _emit_action(
    adapter: SSEAdapter, model: str, json_mod: Any, done: bool
) -> dict[str, Any]:
    """Build and serialize an action output from the adapter."""
    if done:
        return adapter.build_chunk(done=True)
    return adapter.build_chunk(done=False)


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
        created: The Unix timestamp from the upstream response. If None,
            the first SSE data line is inspected for a ``created`` field.
        is_chat: If True, use chat completion format; otherwise use
            completions format.

    Yields:
        Ollama-format NDJSON lines, one per SSE event.
    """
    first_line: str | None = None
    if created is None:
        async for line in _read_sse_lines(upstream_response):
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() != "[DONE]":
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, dict) and "created" in parsed:
                            created = parsed["created"]
                    except json.JSONDecodeError:
                        pass
                first_line = line
                break

    adapter = SSEAdapter(model, created, is_chat)
    done_received = False
    content_emitted = False

    try:
        if first_line is not None:
            action = adapter.process_line(first_line)
            if action == "emit":
                content_emitted = True
                yield (
                    json.dumps(adapter.build_chunk(done=False), separators=(",", ":"))
                    + "\n"
                )
            elif action == "done":
                done_received = True
                yield (
                    json.dumps(adapter.build_chunk(done=True), separators=(",", ":"))
                    + "\n"
                )
            elif action == "error":
                done_received = True
                yield (
                    json.dumps(
                        {
                            "model": model,
                            "error": adapter.last_error or "upstream error in stream",
                            "done": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            elif action == "skip":
                logger.warning("Skipping malformed SSE line: %s", first_line[:100])

        async for line in _read_sse_lines(upstream_response):
            action = adapter.process_line(line)

            if action == "emit":
                content_emitted = True
                yield (
                    json.dumps(adapter.build_chunk(done=False), separators=(",", ":"))
                    + "\n"
                )

            elif action == "done":
                done_received = True
                yield (
                    json.dumps(adapter.build_chunk(done=True), separators=(",", ":"))
                    + "\n"
                )

            elif action == "error":
                done_received = True
                yield (
                    json.dumps(
                        {
                            "model": model,
                            "error": adapter.last_error or "upstream error in stream",
                            "done": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            elif action == "skip":
                logger.warning("Skipping malformed SSE line: %s", line[:100])

    except httpx.ReadTimeout:
        done_received = True
        yield json.dumps(adapter.build_chunk(done=True), separators=(",", ":")) + "\n"
        yield (
            json.dumps(
                {
                    "model": model,
                    "error": "upstream request timed out",
                    "done": True,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    except httpx.StreamError:
        done_received = True
        if not content_emitted and (adapter.content or adapter.tool_calls):
            yield (
                json.dumps(adapter.build_chunk(done=False), separators=(",", ":"))
                + "\n"
            )
        yield (
            json.dumps(
                {
                    "model": model,
                    "error": "upstream connection lost mid-response",
                    "done": True,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    finally:
        if not done_received:
            yield (
                json.dumps(
                    {
                        "model": model,
                        "error": "upstream connection lost mid-response",
                        "done": True,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        await upstream_response.aclose()


async def _read_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Read lines from an SSE response, handling ``\\r\\n`` and ``\\n`` endings."""
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            yield line
    if buffer.strip():
        yield buffer.rstrip("\r")
