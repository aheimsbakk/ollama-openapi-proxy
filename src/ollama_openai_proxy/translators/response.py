"""OpenAI-to-Ollama response translation functions.

Pure functions. No I/O. No framework imports. Each function takes an
OpenAI response body (dict) and returns an Ollama response body (dict).
"""

from __future__ import annotations

import json
import time
from typing import Any


def _ts_to_iso8601(unix_ts: int | float | None) -> str:
    """Convert a Unix timestamp (integer seconds) to an ISO 8601 string."""
    if unix_ts is None:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(unix_ts)))
    except (OSError, ValueError, OverflowError):
        return ""


def completion_to_generate(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/completions response to Ollama /api/generate format."""
    choices = openai_body.get("choices", [])
    usage = openai_body.get("usage", {})

    if not choices:
        return {
            "model": openai_body.get("model", ""),
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }

    choice = choices[0]
    finish_reason = choice.get("finish_reason", "stop")

    result: dict[str, Any] = {
        "model": openai_body.get("model", ""),
        "created_at": _ts_to_iso8601(openai_body.get("created")),
        "response": choice.get("text", ""),
        "done": True,
        "done_reason": finish_reason if finish_reason else "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": 0,
    }

    return result


def chat_completion_to_chat(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/chat/completions response to Ollama /api/chat format."""
    choices = openai_body.get("choices", [])
    usage = openai_body.get("usage", {})

    if not choices:
        return {
            "model": openai_body.get("model", ""),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    # Handle tool calls — OpenAI sends arguments as a JSON string,
    # Ollama expects a parsed JSON object.
    tool_calls = message.get("tool_calls", None)
    if tool_calls:
        tool_calls = _parse_tool_call_arguments(tool_calls)

    content = message.get("content", "") or ""

    result: dict[str, Any] = {
        "model": openai_body.get("model", ""),
        "created_at": _ts_to_iso8601(openai_body.get("created")),
        "message": {
            "role": message.get("role", "assistant"),
            "content": content,
        },
        "done": True,
        "done_reason": finish_reason if finish_reason else "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": 0,
    }

    if tool_calls:
        result["message"]["tool_calls"] = tool_calls

    return result


def _parse_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse the arguments field in tool calls from JSON string to dict."""
    result = []
    for tc in tool_calls:
        parsed = dict(tc)
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            parsed["function"] = {**func, "arguments": json.loads(args_str)}
        except (json.JSONDecodeError, TypeError):
            parsed["function"] = {**func, "arguments": {}}
        result.append(parsed)
    return result


def embeddings_to_embed(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/embeddings response to Ollama /api/embed format."""
    data = openai_body.get("data", [])
    usage = openai_body.get("usage", {})

    embeddings = [item["embedding"] for item in data if "embedding" in item]

    result: dict[str, Any] = {
        "model": openai_body.get("model", ""),
        "embeddings": embeddings,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }

    return result


def embeddings_legacy_to_embeddings(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/embeddings response to Ollama /api/embeddings (legacy) format.

    The legacy endpoint returns a flat ``embedding`` list instead of ``embeddings`` array.
    """
    data = openai_body.get("data", [])
    usage = openai_body.get("usage", {})

    embedding = data[0]["embedding"] if data and "embedding" in data[0] else []

    result: dict[str, Any] = {
        "model": openai_body.get("model", ""),
        "embedding": embedding,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
    }

    return result


def models_list_to_tags(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/models response to Ollama /api/tags format."""
    data = openai_body.get("data", [])
    models = []
    for item in data:
        model_id = item.get("id", "")
        created = item.get("created")
        family = _extract_family(model_id)

        # Populate capabilities from architecture (present on all upstream models).
        arch = item.get("architecture", {})
        capabilities = _extract_capabilities(arch)

        # Populate size and metadata from meta (only present on loaded models).
        meta = item.get("meta", {})

        entry: dict[str, Any] = {
            "name": model_id,
            "model": model_id,
            "modified_at": _ts_to_iso8601(created),
            "size": meta.get("size", 0),
            "digest": _simple_hash(model_id),
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": family,
                "families": [family],
                "parameter_size": _format_parameter_size(meta.get("n_params")),
                "quantization_level": meta.get("ftype", ""),
            },
        }
        if capabilities:
            entry["details"]["capabilities"] = capabilities
        models.append(entry)

    return {"models": models}


def _extract_ctx_from_args(args: list[str]) -> int:
    """Extract context size from llama.cpp ``--ctx-size`` arg list.

    Returns the value following ``--ctx-size`` in the args array,
    or 0 if not found.
    """
    for i, arg in enumerate(args):
        if arg == "--ctx-size" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except (ValueError, TypeError):
                return 0
    return 0


def models_show_to_show(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI single model response to Ollama /api/show format."""
    model_id = openai_body.get("id", "")
    created = openai_body.get("created")
    owned_by = openai_body.get("owned_by", "")
    family = _extract_family(model_id)

    meta = openai_body.get("meta", {})
    arch = openai_body.get("architecture", {})
    status = openai_body.get("status", {})
    capabilities = _extract_capabilities(arch)

    # Build model_info from available GGUF metadata.
    # meta is only present for loaded models; fall back to status.args for
    # context length when meta is not available.
    model_info: dict[str, Any] = {}
    ctx_length = meta.get("n_ctx", 0)
    if not ctx_length:
        ctx_length = _extract_ctx_from_args(status.get("args", []))

    if meta or ctx_length:
        model_info["general.architecture"] = family
        model_info["general.name"] = model_id
        model_info["general.parameter_count"] = meta.get("n_params", 0)
        model_info["general.file_type"] = meta.get("ftype", "")
        model_info["general.size"] = meta.get("size", 0)
        model_info["general.vocab_size"] = meta.get("n_vocab", 0)
        model_info["llm.context_length"] = ctx_length
        model_info["llm.embedding_length"] = meta.get("n_embd", 0)

    # Use upstream's preset string as a close equivalent to a Modelfile.
    modelfile = status.get("preset", "")

    result: dict[str, Any] = {
        "model": model_id,
        "modified_at": _ts_to_iso8601(created),
        "template": "",
        "modelfile": modelfile,
        "parameters": "",
        "model_info": model_info,
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": meta.get("n_params", 0),
            "quantization_level": meta.get("ftype", ""),
        },
    }

    if owned_by:
        result["details"]["parent_model"] = owned_by
    if capabilities:
        result["details"]["capabilities"] = capabilities
    # Add projector_info placeholder for multimodal models.
    if "image" in arch.get("input_modalities", []):
        result["projector_info"] = {}

    return result


def models_list_to_ps(openai_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI /v1/models response to Ollama /api/ps format.

    Equivalent to /api/tags but adds ``expires_at`` and ``size_vram`` fields.
    """
    tags = models_list_to_tags(openai_body)
    for model in tags.get("models", []):
        model["expires_at"] = "2099-12-31T23:59:59Z"
        model["size_vram"] = 0
    return tags


def _extract_capabilities(arch: dict[str, Any]) -> list[str]:
    """Extract model capabilities from the upstream architecture descriptor.

    Returns a list of capability strings (e.g. ``["tools", "vision"]``).
    Returns an empty list when no architecture data is available.
    """
    if not arch:
        return []
    caps: list[str] = []
    output_mods = arch.get("output_modalities", [])
    input_mods = arch.get("input_modalities", [])
    # Text-output chat models are assumed to support tools.
    if "text" in output_mods:
        caps.append("tools")
    # Multimodal models with image input support vision.
    if "image" in input_mods:
        caps.append("vision")
    return caps


def _format_parameter_size(n_params: int | None) -> str:
    """Format a parameter count into a human-readable string like ``"25.2B"``."""
    if n_params is None or n_params <= 0:
        return ""
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.1f}B"
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.1f}M"
    return str(n_params)


def _extract_family(model_id: str) -> str:
    """Extract the model family from a model ID.

    Takes the first segment before any ``-`` or ``:`` character.
    Falls back to ``"unknown"``.
    """
    if not model_id:
        return "unknown"
    # Take the first segment before - or :
    family = model_id.split("-")[0].split(":")[0]
    return family if family else "unknown"


def _simple_hash(value: str) -> str:
    """Generate a deterministic non-cryptographic hash for a model ID.

    Uses Python's built-in hash with a fixed seed for reproducibility.
    Returns a short hex string.
    """
    if not value:
        return "0" * 16
    h = hash(value) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")
