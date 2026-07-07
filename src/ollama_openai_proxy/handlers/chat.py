"""Handler for POST /api/chat.

Pipeline:
  1. Parse Ollama request body
  2. Translate to OpenAI chat completions request
  3. Call upstream (streaming or non-streaming)
  4. Translate response back to Ollama format
  5. Return Ollama response
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.dependencies import get_upstream_client
from ollama_openai_proxy.translators import (
    request as req_trans,
    response as resp_trans,
    streaming,
)

logger = logging.getLogger("ollama_openai_proxy")


async def handle_chat(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse | StreamingResponse:
    """Handle POST /api/chat requests."""
    # Parse the Ollama request body
    try:
        ollama_body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid request body: could not parse JSON")

    # Validate required fields
    if "model" not in ollama_body:
        raise HTTPException(status_code=400, detail="missing required field: model")
    if "messages" not in ollama_body:
        raise HTTPException(status_code=400, detail="missing required field: messages")

    # Translate to OpenAI request
    openai_body = req_trans.chat_to_chat_completions(ollama_body)
    is_streaming = openai_body.get("stream", False)
    model = ollama_body.get("model", "")

    if is_streaming:
        return await _handle_chat_stream(ollama_body, openai_body, model, client)
    else:
        return await _handle_chat_non_stream(openai_body, model, client)


async def _handle_chat_non_stream(
    openai_body: dict[str, Any],
    model: str,
    client: UpstreamClient,
) -> JSONResponse:
    """Handle non-streaming /api/chat requests."""
    upstream_url = f"{client._client.base_url}/chat/completions"
    openai_response = await client.post(upstream_url, json=openai_body)
    ollama_response = resp_trans.chat_completion_to_chat(openai_response)
    return JSONResponse(content=ollama_response)


async def _handle_chat_stream(
    ollama_body: dict[str, Any],
    openai_body: dict[str, Any],
    model: str,
    client: UpstreamClient,
) -> StreamingResponse:
    """Handle streaming /api/chat requests."""
    upstream_url = f"{client._client.base_url}/chat/completions"
    upstream_response = await client.stream_post(upstream_url, json=openai_body)

    return StreamingResponse(
        streaming.sse_to_ollama_stream(
            upstream_response=upstream_response,
            model=model,
            created=None,
            is_chat=True,
        ),
        media_type="application/x-ndjson",
    )
