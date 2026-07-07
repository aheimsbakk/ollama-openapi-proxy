"""Additional tests for request translation edge cases."""

from __future__ import annotations

from typing import Any

import pytest

from ollama_openai_proxy.translators import request as req_trans


class TestGenerateSuffix:
    """Tests for suffix passthrough in generate_to_completion."""

    def test_suffix_passthrough(self) -> None:
        """Suffix field is forwarded to OpenAI."""
        ollama_body = {
            "model": "test",
            "prompt": "hello",
            "suffix": "world",
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["suffix"] == "world"

    def test_no_suffix(self) -> None:
        """Absence of suffix does not add it to the body."""
        ollama_body = {"model": "test", "prompt": "hello"}
        result = req_trans.generate_to_completion(ollama_body)
        assert "suffix" not in result


class TestGenerateSystemList:
    """Tests for system prompt as a list."""

    def test_system_as_list(self) -> None:
        """System prompt as a list is joined with newlines."""
        ollama_body = {
            "model": "test",
            "prompt": "hello",
            "system": ["Be concise.", "No fluff."],
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["prompt"] == "Be concise.\nNo fluff."

    def test_system_as_list_empty_prompt(self) -> None:
        """System as list with empty prompt replaces the prompt."""
        ollama_body = {
            "model": "test",
            "prompt": "",
            "system": ["Be concise."],
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["prompt"] == "Be concise."

    def test_system_not_string_or_list(self) -> None:
        """Non-string, non-list system is silently ignored."""
        ollama_body = {
            "model": "test",
            "prompt": "hello",
            "system": 42,
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["prompt"] == "hello"


class TestGenerateImages:
    """Tests for image handling in generate_to_completion."""

    def test_images_with_system(self) -> None:
        """Images with an existing system prompt prepend both."""
        ollama_body = {
            "model": "test",
            "prompt": "what is this",
            "system": "You are a vision assistant.",
            "images": ["base64img1"],
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert "You are a vision assistant." in result["prompt"]
        assert "[Image attached]" in result["prompt"]
        assert "what is this" in result["prompt"]

    def test_images_without_system(self) -> None:
        """Images without a system prompt prepend only the image marker."""
        ollama_body = {
            "model": "test",
            "prompt": "what is this",
            "images": ["base64img1"],
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert "[Image attached]" in result["prompt"]
        assert result["prompt"].startswith("[Image attached]")

    def test_empty_images_list(self) -> None:
        """An empty images list is ignored."""
        ollama_body = {
            "model": "test",
            "prompt": "hello",
            "images": [],
        }
        result = req_trans.generate_to_completion(ollama_body)
        assert result["prompt"] == "hello"
        assert "Image attached" not in result["prompt"]


class TestChatFormatSchema:
    """Tests for format-as-schema in chat translation."""

    def test_chat_format_schema(self) -> None:
        """A dict format is mapped to response_format json_schema."""
        ollama_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "format": {"type": "object", "properties": {"answer": {"type": "string"}}},
        }
        result = req_trans.chat_to_chat_completions(ollama_body)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        }


class TestTranslateMessages:
    """Tests for _translate_messages edge cases."""

    def test_images_without_string_content(self) -> None:
        """Images with non-string content produce only image parts."""
        messages = [{"role": "user", "content": 123, "images": ["img1"]}]
        result = req_trans._translate_messages(messages)
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "image_url"
        assert "img1" in result[0]["content"][0]["image_url"]["url"]

    def test_tool_name_absorption_string(self) -> None:
        """Tool name is absorbed into content for tool role with string content."""
        messages = [
            {
                "role": "tool",
                "content": "result data",
                "tool_name": "get_weather",
            }
        ]
        result = req_trans._translate_messages(messages)
        assert result[0]["content"] == "Tool: get_weather\nresult data"

    def test_tool_name_absorption_list_content(self) -> None:
        """Tool name is absorbed into content for tool role with list content."""
        messages = [
            {
                "role": "tool",
                "content": [{"type": "text", "text": "parsed result"}],
                "tool_name": "get_weather",
            }
        ]
        result = req_trans._translate_messages(messages)
        expected = "Tool: get_weather\n" + str(
            [{"type": "text", "text": "parsed result"}]
        )
        assert result[0]["content"] == expected

    def test_tool_name_without_tool_role(self) -> None:
        """Tool name on non-tool role is not absorbed."""
        messages = [{"role": "user", "content": "hi", "tool_name": "get_weather"}]
        result = req_trans._translate_messages(messages)
        assert result[0]["content"] == "hi"
        assert "Tool:" not in result[0]["content"]

    def test_tool_calls_passthrough(self) -> None:
        """Tool calls in messages pass through unchanged."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        messages = [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
        result = req_trans._translate_messages(messages)
        assert result[0]["tool_calls"] == tool_calls

    def test_message_with_no_content(self) -> None:
        """A message with no content field defaults to empty string."""
        messages = [{"role": "assistant"}]
        result = req_trans._translate_messages(messages)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""

    def test_message_with_none_content(self) -> None:
        """A message with explicit None content omits the field."""
        messages = [{"role": "assistant", "content": None}]
        result = req_trans._translate_messages(messages)
        assert result[0]["role"] == "assistant"
        assert "content" not in result[0]


class TestEmbedDimensions:
    """Tests for embed_to_embeddings with dimensions option."""

    def test_dimensions_option(self) -> None:
        """Dimensions from options are forwarded."""
        ollama_body = {
            "model": "all-minilm",
            "input": "hello",
            "options": {"dimensions": 384},
        }
        result = req_trans.embed_to_embeddings(ollama_body)
        assert result["dimensions"] == 384

    def test_no_dimensions_option(self) -> None:
        """Absence of dimensions does not add the field."""
        ollama_body = {"model": "all-minilm", "input": "hello"}
        result = req_trans.embed_to_embeddings(ollama_body)
        assert "dimensions" not in result

    def test_options_not_dict(self) -> None:
        """Non-dict options are ignored."""
        ollama_body = {
            "model": "all-minilm",
            "input": "hello",
            "options": "invalid",
        }
        result = req_trans.embed_to_embeddings(ollama_body)
        assert "dimensions" not in result
