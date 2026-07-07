"""SSE-to-NDJSON stream adapter.

Reads SSE events from an upstream OpenAI response and yields Ollama-format
NDJSON lines. Implements the state machine described in BLUEPRINT.md
Appendix B.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from ollama_openai_proxy.translators.streaming_adapter import SSEAdapter

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
        created: The Unix timestamp from the upstream response. If None,
            the first SSE data line is inspected for a ``created`` field.
        is_chat: If True, use chat completion format; otherwise use
            completions format.

    Yields:
        Ollama-format NDJSON lines, one per SSE event.
    """
    adapter = SSEAdapter(model, created, is_chat)
    done_received = False
    content_emitted = False

    logger.info(
        "Stream start — model=%s is_chat=%s",
        model,
        is_chat,
    )
    line_count = 0

    try:
        async for line in _read_sse_lines(upstream_response):
            if not line.startswith("data: "):
                continue

            data = line[6:].strip()
            if data == "[DONE]":
                done_received = True
                logger.debug("Stream — [DONE] received after %d line(s)", line_count)
                yield (
                    json.dumps(adapter.build_chunk(done=True), separators=(",", ":"))
                    + "\n"
                )
                break

            # Extract created timestamp from the first data line if not provided.
            if created is None:
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict) and "created" in parsed:
                        adapter.created_at = _ts_to_iso8601(parsed["created"])
                        created = parsed["created"]
                except json.JSONDecodeError:
                    pass

            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Stream — skipping malformed SSE line: %s", line[:100])
                continue

            if isinstance(parsed, dict) and "error" in parsed:
                done_received = True
                logger.warning("Stream — upstream error in chunk: %s", parsed["error"])
                yield (
                    json.dumps(
                        {"error": parsed["error"]},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                break

            content_emitted = True
            line_count += 1
            adapter._accumulate(parsed)

            chunk = adapter.build_chunk(done=False)
            logger.debug(
                "Stream — line %d: chunk=%s", line_count, _truncate_json(chunk)
            )
            yield json.dumps(chunk, separators=(",", ":")) + "\n"

    except httpx.ReadTimeout:
        done_received = True
        logger.error("Stream — upstream read timeout after %d line(s)", line_count)
        yield json.dumps(adapter.build_chunk(done=True), separators=(",", ":")) + "\n"
        yield (
            json.dumps(
                {"error": "request to AI server timed out"},
                separators=(",", ":"),
            )
            + "\n"
        )

    except httpx.StreamError:
        done_received = True
        logger.error("Stream — upstream connection lost after %d line(s)", line_count)
        if not content_emitted and (adapter.content or adapter.tool_calls):
            yield (
                json.dumps(adapter.build_chunk(done=False), separators=(",", ":"))
                + "\n"
            )
        yield (
            json.dumps(
                {"error": "connection to AI server lost during response"},
                separators=(",", ":"),
            )
            + "\n"
        )

    finally:
        if not done_received:
            logger.warning("Stream — connection closed without [DONE]")
            yield (
                json.dumps(
                    {"error": "connection to AI server lost during response"},
                    separators=(",", ":"),
                )
                + "\n"
            )
        else:
            logger.info("Stream — completed after %d line(s)", line_count)
        await upstream_response.aclose()


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


def _truncate_json(obj: dict[str, Any], max_len: int = 200) -> str:
    """Truncate a dict to max_len characters for logging."""
    text = json.dumps(obj, separators=(",", ":"))
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# Backward-compatible re-exports for tests.
_SSEAdapter = SSEAdapter
