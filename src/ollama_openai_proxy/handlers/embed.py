"""Handler for POST /api/embed and POST /api/embeddings (legacy).

Both endpoints map to OpenAI /v1/embeddings with different input formats.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.dependencies import get_upstream_client
from ollama_openai_proxy.translators import request as req_trans, response as resp_trans

logger = logging.getLogger("ollama_openai_proxy")


async def handle_embed(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle POST /api/embed requests."""
    ollama_body = await _parse_body(request)
    openai_body = req_trans.embed_to_embeddings(ollama_body)
    upstream_url = f"{client._client.base_url}/embeddings"
    openai_response = await client.post(upstream_url, json=openai_body)
    ollama_response = resp_trans.embeddings_to_embed(openai_response)
    return JSONResponse(content=ollama_response)


async def handle_embeddings_legacy(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle POST /api/embeddings (legacy) requests."""
    ollama_body = await _parse_body(request)
    openai_body = req_trans.embeddings_legacy_to_embeddings(ollama_body)
    upstream_url = f"{client._client.base_url}/embeddings"
    openai_response = await client.post(upstream_url, json=openai_body)
    ollama_response = resp_trans.embeddings_legacy_to_embeddings(openai_response)
    return JSONResponse(content=ollama_response)


async def _parse_body(request: Request) -> dict[str, Any]:
    """Parse and validate the request body."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid request body: could not parse JSON")

    if "model" not in body:
        raise HTTPException(status_code=400, detail="missing required field: model")
    if "input" not in body and "prompt" not in body:
        raise HTTPException(status_code=400, detail="missing required field: input")

    return body
