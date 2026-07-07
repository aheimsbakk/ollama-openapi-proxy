"""Handler for GET /api/tags, POST /api/show, GET /api/ps, and GET /api/version.

Informational endpoints that fetch model metadata from the upstream server.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ollama_openai_proxy import __version__
from ollama_openai_proxy.client import UpstreamClient
from ollama_openai_proxy.dependencies import get_upstream_client
from ollama_openai_proxy.translators import response as resp_trans

logger = logging.getLogger("ollama_openai_proxy")


async def handle_tags(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle GET /api/tags — list available models."""
    logger.info("GET /api/tags — listing models")
    upstream_url = f"{client.base_url}/models"
    openai_response = await client.get(upstream_url)
    ollama_response = resp_trans.models_list_to_tags(openai_response)
    count = len(ollama_response.get("models", []))
    logger.info("GET /api/tags — returning %d models", count)
    return JSONResponse(content=ollama_response)


async def handle_show(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle POST /api/show — show model details.

    Fetches the model list from the upstream and finds the requested model
    by ID. This works with upstream servers (like llama.cpp) that do not
    expose a per-model endpoint at ``/v1/models/{id}``.
    """
    logger.info("POST /api/show — showing model details")

    try:
        body = await request.json()
    except Exception:
        logger.warning("POST /api/show — invalid JSON body")
        raise HTTPException(
            status_code=400, detail="Invalid request body. Could not parse JSON."
        )

    if "model" not in body:
        logger.warning("POST /api/show — missing required field: model")
        raise HTTPException(status_code=400, detail="missing required field: model")

    model_name = body["model"]
    logger.debug("POST /api/show — model=%s", model_name)
    upstream_url = f"{client.base_url}/models"
    openai_list = await client.get(upstream_url)
    data = openai_list.get("data", [])
    for item in data:
        if item.get("id") == model_name:
            ollama_response = resp_trans.models_show_to_show(item)
            logger.info("POST /api/show — model '%s' found", model_name)
            return JSONResponse(content=ollama_response)

    logger.warning("POST /api/show — model '%s' not found on upstream", model_name)
    raise HTTPException(
        status_code=404, detail=f"Model '{model_name}' not found on the AI server."
    )


async def handle_ps(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle GET /api/ps — show running models."""
    logger.info("GET /api/ps — listing running models")
    upstream_url = f"{client.base_url}/models"
    openai_response = await client.get(upstream_url)
    ollama_response = resp_trans.models_list_to_ps(openai_response)
    count = len(ollama_response.get("models", []))
    logger.info("GET /api/ps — returning %d models", count)
    return JSONResponse(content=ollama_response)


def handle_version(
    request: Request,
    client: UpstreamClient = Depends(get_upstream_client),
) -> JSONResponse:
    """Handle GET /api/version — return program version."""
    logger.info("GET /api/version — returning version %s", __version__)
    return JSONResponse(content={"version": __version__})
