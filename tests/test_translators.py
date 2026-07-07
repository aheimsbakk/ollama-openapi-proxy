"""Tests for the request and response translation functions."""

from __future__ import annotations

from typing import Any

from ollama_openai_proxy.translators import request as req_trans, response as resp_trans


class TestGenerateToCompletion:
    """Tests for generate_to_completion."""

    def test_basic_translation(self) -> None:
        ollama_body = {
            "model": "llama-3.2-3b",
            "prompt": "Why is the sky blue?",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "seed": 42,
                "num_predict": 100,
            },
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["model"] == "llama-3.2-3b"
        assert result["prompt"] == "Why is the sky blue?"
        assert result["temperature"] == 0.7
        assert result["seed"] == 42
        assert result["max_tokens"] == 100
        assert result["stream"] is False
        assert "top_k" not in result  # dropped
        assert "num_ctx" not in result  # dropped

    def test_format_json(self) -> None:
        ollama_body = {"model": "test", "prompt": "hi", "format": "json"}
        result = req_trans.generate_to_completion(ollama_body)
        assert result["response_format"] == {"type": "json_object"}

    def test_format_schema(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        ollama_body = {"model": "test", "prompt": "hi", "format": schema}
        result = req_trans.generate_to_completion(ollama_body)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": schema,
        }

    def test_system_prompt_prepend(self) -> None:
        ollama_body = {
            "model": "test",
            "prompt": "hello",
            "system": "You are a helpful assistant.",
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert "You are a helpful assistant." in result["prompt"]
        assert "hello" in result["prompt"]


class TestChatToChatCompletions:
    """Tests for chat_to_chat_completions."""

    def test_basic_translation(self) -> None:
        ollama_body = {
            "model": "llama-3.2-3b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        result = req_trans.chat_to_chat_completions(ollama_body)
        assert result["model"] == "llama-3.2-3b"
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["stream"] is False

    def test_messages_with_images(self) -> None:
        ollama_body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "what is this", "images": ["base64data"]}
            ],
        }
        result = req_trans.chat_to_chat_completions(ollama_body)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        text_part = msg["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "what is this"
        image_part = msg["content"][1]
        assert image_part["type"] == "image_url"
        assert "base64data" in image_part["image_url"]["url"]

    def test_tools_passthrough(self) -> None:
        tools = [{"type": "function", "function": {"name": "test"}}]
        ollama_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
        }
        result = req_trans.chat_to_chat_completions(ollama_body)
        assert result["tools"] == tools


class TestCompletionToGenerate:
    """Tests for completion_to_generate."""

    def test_basic_translation(self) -> None:
        openai_body = {
            "id": "cmpl-abc123",
            "created": 1720000000,
            "model": "llama-3.2-3b",
            "choices": [{"text": "The sky appears blue.", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 5},
        }
        result = resp_trans.completion_to_generate(openai_body)
        assert result["model"] == "llama-3.2-3b"
        assert result["response"] == "The sky appears blue."
        assert result["done"] is True
        assert result["done_reason"] == "stop"
        assert result["prompt_eval_count"] == 8
        assert result["eval_count"] == 5
        assert "2024-07-03" in result["created_at"]

    def test_empty_choices(self) -> None:
        openai_body = {"model": "test", "choices": []}
        result = resp_trans.completion_to_generate(openai_body)
        assert result["done"] is True
        assert result["response"] == ""


class TestChatCompletionToChat:
    """Tests for chat_completion_to_chat."""

    def test_basic_translation(self) -> None:
        openai_body = {
            "id": "chatcmpl-xyz",
            "created": 1720000002,
            "model": "llama-3.2-3b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        result = resp_trans.chat_completion_to_chat(openai_body)
        assert result["model"] == "llama-3.2-3b"
        assert result["message"]["content"] == "Hello world"
        assert result["done"] is True
        assert result["done_reason"] == "stop"

    def test_tool_calls_translation(self) -> None:
        openai_body = {
            "id": "chatcmpl-xyz",
            "created": 1720000002,
            "model": "llama-3.2-3b",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 15},
        }
        result = resp_trans.chat_completion_to_chat(openai_body)
        assert result["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["message"]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo"
        }
        assert result["done_reason"] == "tool_calls"


class TestEmbeddingsToEmbed:
    """Tests for embeddings_to_embed."""

    def test_basic_translation(self) -> None:
        openai_body = {
            "model": "all-minilm",
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ],
            "usage": {"prompt_tokens": 6},
        }
        result = resp_trans.embeddings_to_embed(openai_body)
        assert result["model"] == "all-minilm"
        assert result["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result["prompt_eval_count"] == 6


class TestModelsListToTags:
    """Tests for models_list_to_tags."""

    def test_basic_translation(self) -> None:
        openai_body = {
            "data": [
                {"id": "llama-3.2-3b", "created": 1720000000, "owned_by": "library"},
            ],
        }
        result = resp_trans.models_list_to_tags(openai_body)
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "llama-3.2-3b"
        assert result["models"][0]["details"]["format"] == "gguf"
        assert result["models"][0]["details"]["family"] == "llama"
        assert result["models"][0]["size"] == 0


class TestExtractFamily:
    """Tests for the _extract_family helper."""

    def test_llama_family(self) -> None:
        assert resp_trans._extract_family("llama-3.2-3b") == "llama"

    def test_mistral_family(self) -> None:
        assert resp_trans._extract_family("mistral-7b") == "mistral"

    def test_empty_string(self) -> None:
        assert resp_trans._extract_family("") == "unknown"


class TestSimpleHash:
    """Tests for the _simple_hash helper."""

    def test_deterministic(self) -> None:
        h1 = resp_trans._simple_hash("test-model")
        h2 = resp_trans._simple_hash("test-model")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        assert resp_trans._simple_hash("model-a") != resp_trans._simple_hash("model-b")

    def test_empty_string(self) -> None:
        assert resp_trans._simple_hash("") == "0" * 16
        assert resp_trans._simple_hash("model-a") == resp_trans._simple_hash("model-a")


class TestExtractCapabilities:
    """Tests for the _extract_capabilities helper."""

    def test_text_only_gets_tools(self) -> None:
        arch = {"output_modalities": ["text"], "input_modalities": ["text"]}
        assert resp_trans._extract_capabilities(arch) == ["tools"]

    def test_vision_adds_vision_capability(self) -> None:
        arch = {
            "output_modalities": ["text"],
            "input_modalities": ["text", "image"],
        }
        caps = resp_trans._extract_capabilities(arch)
        assert "tools" in caps
        assert "vision" in caps

    def test_no_architecture_returns_empty(self) -> None:
        assert resp_trans._extract_capabilities({}) == []

    def test_non_text_output_no_capabilities(self) -> None:
        arch = {"output_modalities": ["embedding"], "input_modalities": ["text"]}
        assert resp_trans._extract_capabilities(arch) == []


class TestFormatParameterSize:
    """Tests for the _format_parameter_size helper."""

    def test_none_returns_empty(self) -> None:
        assert resp_trans._format_parameter_size(None) == ""

    def test_zero_returns_empty(self) -> None:
        assert resp_trans._format_parameter_size(0) == ""

    def test_billions_formatted_as_B(self) -> None:
        assert resp_trans._format_parameter_size(25_233_142_046) == "25.2B"

    def test_millions_formatted_as_M(self) -> None:
        assert resp_trans._format_parameter_size(3_000_000) == "3.0M"

    def test_small_numbers_returned_as_is(self) -> None:
        assert resp_trans._format_parameter_size(500) == "500"


class TestModelsListToTagsCapabilities:
    """models_list_to_tags populates capabilities from architecture."""

    def test_vision_model_tags(self) -> None:
        body = {
            "data": [
                {
                    "id": "gemma-4",
                    "created": 1720000000,
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "meta": {
                        "n_params": 25_233_142_046,
                        "size": 16_995_155_064,
                        "ftype": "Q4_K - Medium",
                    },
                },
            ],
        }
        result = resp_trans.models_list_to_tags(body)
        model = result["models"][0]
        assert model["name"] == "gemma-4"
        assert model["size"] == 16_995_155_064
        assert model["details"]["parameter_size"] == "25.2B"
        assert model["details"]["quantization_level"] == "Q4_K - Medium"
        assert model["details"]["capabilities"] == ["tools", "vision"]

    def test_text_only_model_tags(self) -> None:
        body = {
            "data": [
                {
                    "id": "llama-3.2-3b",
                    "created": 1720000000,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                },
            ],
        }
        result = resp_trans.models_list_to_tags(body)
        model = result["models"][0]
        assert model["details"]["capabilities"] == ["tools"]
        assert "vision" not in model["details"]["capabilities"]
        assert model["size"] == 0  # no meta

    def test_no_architecture_defaults_no_capabilities_key(self) -> None:
        """When architecture is absent, capabilities key is not set."""
        body = {
            "data": [
                {"id": "test-model", "created": 1720000000},
            ],
        }
        result = resp_trans.models_list_to_tags(body)
        assert "capabilities" not in result["models"][0]["details"]


class TestModelsShowToShow:
    """Tests for models_show_to_show with upstream metadata."""

    def test_with_full_upstream_data(self) -> None:
        body = {
            "id": "gemma-4",
            "created": 1783455247,
            "owned_by": "llamacpp",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "meta": {
                "n_params": 25_233_142_046,
                "n_ctx": 256_000,
                "n_embd": 2816,
                "n_vocab": 262_144,
                "size": 16_995_155_064,
                "ftype": "Q4_K - Medium",
            },
            "status": {
                "preset": "[gemma-4]\ntemperature = 1.0\n",
            },
        }
        result = resp_trans.models_show_to_show(body)
        assert result["model"] == "gemma-4"
        assert result["modelfile"] == "[gemma-4]\ntemperature = 1.0\n"
        assert "tools" in result["details"]["capabilities"]
        assert "vision" in result["details"]["capabilities"]
        assert result["model_info"]["general.parameter_count"] == 25_233_142_046
        assert result["model_info"]["general.file_type"] == "Q4_K - Medium"
        assert result["model_info"]["llm.context_length"] == 256_000
        assert result["model_info"]["llm.embedding_length"] == 2816
        assert result["model_info"]["general.vocab_size"] == 262_144
        assert result["model_info"]["general.size"] == 16_995_155_064
        assert result["projector_info"] == {}

    def test_minimal_data(self) -> None:
        body = {"id": "test-model", "created": 1720000000}
        result = resp_trans.models_show_to_show(body)
        assert result["model"] == "test-model"
        assert result["model_info"] == {}
        assert result["modelfile"] == ""
        assert "capabilities" not in result.get("details", {})

    def test_context_length_from_args_fallback(self) -> None:
        """Unloaded model gets context_length from status.args --ctx-size."""
        body = {
            "id": "qwen-3.6-think-coding",
            "created": 1783455247,
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "status": {
                "args": [
                    "/app/llama-server",
                    "--ctx-size",
                    "256000",
                    "--alias",
                    "qwen-3.6-think-coding",
                ],
            },
        }
        result = resp_trans.models_show_to_show(body)
        assert result["model_info"]["llm.context_length"] == 256_000
        assert result["model_info"]["general.architecture"] == "qwen"
        assert result["model_info"]["general.name"] == "qwen-3.6-think-coding"

    def test_meta_takes_precedence_over_args(self) -> None:
        """When both meta.n_ctx and status.args are present, meta wins."""
        body = {
            "id": "test-model",
            "created": 1783455247,
            "meta": {"n_ctx": 128_000},
            "status": {"args": ["--ctx-size", "999999"]},
        }
        result = resp_trans.models_show_to_show(body)
        assert result["model_info"]["llm.context_length"] == 128_000


class TestExtractCtxFromArgs:
    """Tests for the _extract_ctx_from_args helper."""

    def test_ctx_size_found(self) -> None:
        args = ["--ctx-size", "256000", "--alias", "test"]
        assert resp_trans._extract_ctx_from_args(args) == 256_000

    def test_no_ctx_size(self) -> None:
        args = ["--alias", "test", "--temperature", "1.0"]
        assert resp_trans._extract_ctx_from_args(args) == 0

    def test_empty_args(self) -> None:
        assert resp_trans._extract_ctx_from_args([]) == 0

    def test_non_numeric_value(self) -> None:
        args = ["--ctx-size", "not-a-number"]
        assert resp_trans._extract_ctx_from_args(args) == 0
