"""Ollama-to-OpenAI request translation functions.

Pure functions. No I/O. No framework imports. Each function takes an
Ollama request body (dict) and returns an OpenAI request body (dict).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ollama_openai_proxy")


def generate_to_completion(ollama_body: dict[str, Any]) -> dict[str, Any]:
    """Translate a POST /api/generate request to OpenAI /v1/completions."""
    body: dict[str, Any] = {
        "model": ollama_body.get("model", ""),
        "prompt": ollama_body.get("prompt", ""),
    }
    logger.debug(
        "translate generate_to_completion — input model=%s",
        ollama_body.get("model", ""),
    )

    # Suffix (rarely used but supported by OpenAI completions)
    if "suffix" in ollama_body:
        body["suffix"] = ollama_body["suffix"]
        logger.debug("translate generate_to_completion — suffix present")

    # System prompt — prepend to the prompt string
    if "system" in ollama_body:
        system = ollama_body["system"]
        if isinstance(system, str) and body["prompt"]:
            body["prompt"] = f"{system}\n\n{body['prompt']}"
        elif isinstance(system, str):
            body["prompt"] = system
        elif isinstance(system, list):
            body["prompt"] = "\n".join(str(s) for s in system)
        logger.debug("translate generate_to_completion — system prompt prepended")

    # Images — prepend as a system message with image content
    if "images" in ollama_body and ollama_body["images"]:
        image_parts = []
        for img in ollama_body["images"]:
            image_parts.append(f"data:image/png;base64,{img}")
        if image_parts:
            if "system" in ollama_body:
                body["prompt"] = (
                    f"{ollama_body['system']}\n\n[Image attached]" + body["prompt"]
                )
            else:
                body["prompt"] = "[Image attached]" + body["prompt"]
        logger.debug(
            "translate generate_to_completion — %d image(s) attached",
            len(ollama_body["images"]),
        )

    # Format → response_format
    if "format" in ollama_body:
        fmt = ollama_body["format"]
        if fmt == "json":
            body["response_format"] = {"type": "json_object"}
            logger.debug("translate generate_to_completion — format=json")
        elif isinstance(fmt, dict):
            body["response_format"] = {"type": "json_schema", "json_schema": fmt}
            logger.debug("translate generate_to_completion — format=schema")

    # Options mapping
    options = ollama_body.get("options", {})
    if isinstance(options, dict):
        _map_options_to_completion(options, body)
        if options:
            logger.debug("translate generate_to_completion — options mapped")

    # Stream flag
    body["stream"] = ollama_body.get("stream", False)

    return body


def _map_options_to_completion(options: dict[str, Any], body: dict[str, Any]) -> None:
    """Map Ollama options to OpenAI completions parameters."""
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": None,  # Not supported by OpenAI completions
        "stop": "stop",
        "seed": "seed",
        "num_predict": "max_tokens",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "num_ctx": None,  # Managed by upstream
    }
    for ollama_key, openai_key in mapping.items():
        if ollama_key in options and openai_key is not None:
            body[openai_key] = options[ollama_key]


def chat_to_chat_completions(ollama_body: dict[str, Any]) -> dict[str, Any]:
    """Translate a POST /api/chat request to OpenAI /v1/chat/completions."""
    body: dict[str, Any] = {
        "model": ollama_body.get("model", ""),
        "messages": _translate_messages(ollama_body.get("messages", [])),
    }
    logger.debug(
        "translate chat_to_chat_completions — model=%s messages=%d",
        ollama_body.get("model", ""),
        len(ollama_body.get("messages", [])),
    )

    # Tools pass through directly
    if "tools" in ollama_body:
        body["tools"] = ollama_body["tools"]
        logger.debug(
            "translate chat_to_chat_completions — %d tool(s)", len(ollama_body["tools"])
        )

    # Format → response_format
    if "format" in ollama_body:
        fmt = ollama_body["format"]
        if fmt == "json":
            body["response_format"] = {"type": "json_object"}
            logger.debug("translate chat_to_chat_completions — format=json")
        elif isinstance(fmt, dict):
            body["response_format"] = {"type": "json_schema", "json_schema": fmt}
            logger.debug("translate chat_to_chat_completions — format=schema")

    # Options mapping (same as generate)
    options = ollama_body.get("options", {})
    if isinstance(options, dict):
        _map_options_to_chat(options, body)
        if options:
            logger.debug("translate chat_to_chat_completions — options mapped")

    # Stream flag
    body["stream"] = ollama_body.get("stream", False)

    return body


def _translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Ollama messages to OpenAI messages format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images = msg.get("images", [])

        # Tool calls pass through directly
        tool_calls = msg.get("tool_calls", None)

        # Tool name absorption
        tool_name = msg.get("tool_name", None)

        if images and isinstance(content, str):
            # Convert to content parts array (OpenAI multimodal format)
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
            for img in images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )
            content = content_parts
        elif images:
            # Content is not a string but images are present
            content_parts = []
            for img in images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )
            content = content_parts

        translated: dict[str, Any] = {"role": role}
        if content is not None:
            translated["content"] = content
        if tool_calls is not None:
            translated["tool_calls"] = tool_calls
        if tool_name is not None and role == "tool":
            # Absorb tool_name into content for OpenAI
            if isinstance(content, str):
                translated["content"] = f"Tool: {tool_name}\n{content}"
            elif isinstance(content, list):
                translated["content"] = f"Tool: {tool_name}\n{content}"

        result.append(translated)

    return result


def _map_options_to_chat(options: dict[str, Any], body: dict[str, Any]) -> None:
    """Map Ollama options to OpenAI chat completions parameters."""
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": None,
        "stop": "stop",
        "seed": "seed",
        "num_predict": "max_tokens",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "num_ctx": None,
    }
    for ollama_key, openai_key in mapping.items():
        if ollama_key in options and openai_key is not None:
            body[openai_key] = options[ollama_key]


def embed_to_embeddings(ollama_body: dict[str, Any]) -> dict[str, Any]:
    """Translate a POST /api/embed request to OpenAI /v1/embeddings."""
    body: dict[str, Any] = {
        "model": ollama_body.get("model", ""),
        "input": ollama_body.get("input", ""),
    }
    logger.debug(
        "translate embed_to_embeddings — model=%s",
        ollama_body.get("model", ""),
    )

    # Truncate check — if false and input exceeds context, the handler
    # rejects before forwarding. We don't check length here.

    # Options may contain dimension hints
    options = ollama_body.get("options", {})
    if isinstance(options, dict) and "dimensions" in options:
        body["dimensions"] = options["dimensions"]
        logger.debug(
            "translate embed_to_embeddings — dimensions=%s", options["dimensions"]
        )

    return body


def embeddings_legacy_to_embeddings(ollama_body: dict[str, Any]) -> dict[str, Any]:
    """Translate a POST /api/embeddings (legacy) request to OpenAI /v1/embeddings.

    The legacy endpoint accepts a single ``prompt`` string instead of ``input``.
    """
    logger.debug(
        "translate embeddings_legacy_to_embeddings — model=%s",
        ollama_body.get("model", ""),
    )
    return {
        "model": ollama_body.get("model", ""),
        "input": ollama_body.get("prompt", ""),
    }
