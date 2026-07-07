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
    logger.info("POST /api/embed — handling request")
    ollama_body = await _parse_body(request)
    model = ollama_body.get("model", "")
    logger.debug("POST /api/embed — model=%s", model)
    openai_body = req_trans.embed_to_embeddings(ollama_body)
    upstream_url = f"{client.base_url}/embeddings"
    logger.debug("POST /api/embed — forwarding to upstream: %s", upstream_url)
    openai_response = await client.post(upstream_url, json=openai_body)
    ollama_response = resp_trans.embeddings_to_embed(openai_response)
    logger.info("POST /api/embed — response ready for model=%s", model)
    return JSONResponse(content=ollama_response)


async def handle_embeddings_legacy(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle POST /api/embeddings (legacy) requests."""
    logger.info("POST /api/embeddings — handling legacy request")
    ollama_body = await _parse_body(request)
    model = ollama_body.get("model", "")
    logger.debug("POST /api/embeddings — model=%s", model)
    openai_body = req_trans.embeddings_legacy_to_embeddings(ollama_body)
    upstream_url = f"{client.base_url}/embeddings"
    logger.debug("POST /api/embeddings — forwarding to upstream: %s", upstream_url)
    openai_response = await client.post(upstream_url, json=openai_body)
    ollama_response = resp_trans.embeddings_legacy_to_embeddings(openai_response)
    logger.info("POST /api/embeddings — response ready for model=%s", model)
    return JSONResponse(content=ollama_response)


async def _parse_body(request: Request) -> dict[str, Any]:
    """Parse and validate the request body."""
    try:
        body = await request.json()
    except Exception:
        logger.warning("Embed request — invalid JSON body")
        raise HTTPException(
            status_code=400, detail="Invalid request body. Could not parse JSON."
        )

    if "model" not in body:
        logger.warning("Embed request — missing required field: model")
        raise HTTPException(status_code=400, detail="missing required field: model")
    if "input" not in body and "prompt" not in body:
        logger.warning("Embed request — missing required field: input/prompt")
        raise HTTPException(
            status_code=400, detail="Missing required field: input or prompt."
        )

    return body
