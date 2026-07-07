# Changelog

## [0.2.1] - 2026-07-07

- **why:** Rewrite user-facing error messages and CLI help text in plain language for clarity
- **model:** deepseek-v4-flash-free
- **tags:** refactor, clear-language, error-messages, docs

### Changed

- `src/ollama_openai_proxy/client.py` — replaced "upstream" with "AI server" in 7 user-facing error messages (unreachable, timeout, non-200).
- `src/ollama_openai_proxy/translators/streaming.py` — replaced "upstream" with "AI server" in 4 NDJSON streaming error payloads.
- `src/ollama_openai_proxy/handlers/generate.py`, `chat.py`, `embed.py`, `models.py` — rewritten HTTP 400 error details as full sentences with capital letters and periods.
- `src/ollama_openai_proxy/handlers/models.py` — changed "model not found on upstream" to "Model '...' not found on the AI server." (HTTP 404).
- `src/ollama_openai_proxy/router.py` — shortened unsupported-endpoint message from "endpoint not supported by this proxy" to "Endpoint not supported".
- `src/ollama_openai_proxy/cli.py` — replaced "upstream server", "bind", and passive voice in 4 CLI help texts with plain alternatives.
- `src/ollama_openai_proxy/server.py` — replaced "upstream server" with "AI server" in the FastAPI auto-generated API description.
- `src/ollama_openai_proxy/dependencies.py` — rewritten RuntimeError message to say "HTTP client not initialized" instead of "UpstreamClient not initialized".
- `src/ollama_openai_proxy/errors.py` — trimmed bragging sentence from `global_exception_handler` docstring.
- `src/ollama_openai_proxy/client.py`, `translators/streaming.py` — changed `_truncate_body` and `_truncate_json` docstrings from passive to active voice.
- `tests/test_dependencies.py` — updated assertion to match new RuntimeError message.

## [0.2.0] - 2026-07-07

- **why:** Expose model capabilities, context length, and metadata so clients detect tool/vision support
- **model:** deepseek-v4-flash-free
- **tags:** models, capabilities, context-length, show

### Added

- `src/ollama_openai_proxy/translators/response.py` — `_extract_capabilities` helper reads `output_modalities`/`input_modalities` from upstream architecture to populate `["tools", "vision"]` on discovery endpoints.
- `src/ollama_openai_proxy/translators/response.py` — `_format_parameter_size` helper converts raw `n_params` to human-readable strings (e.g. `"25.2B"`).
- `src/ollama_openai_proxy/translators/response.py` — `_extract_ctx_from_args` parses `--ctx-size` from llama.cpp `status.args` as fallback for unloaded models.
- `src/ollama_openai_proxy/translators/response.py` — `models_show_to_show` now populates `model_info` with `llm.context_length`, `llm.embedding_length`, `general.vocab_size`, `general.size`, and populates `modelfile` from `status.preset`; adds `projector_info` for multimodal models.
- `tests/test_translators.py` — 14 new test cases across `TestExtractCapabilities`, `TestFormatParameterSize`, `TestModelsListToTagsCapabilities`, `TestModelsShowToShow`, `TestExtractCtxFromArgs`.
- `tests/test_models.py` — `test_show_model_not_found` for 404 path.

### Changed

- `src/ollama_openai_proxy/handlers/models.py` — `handle_show` now fetches the model list from `/v1/models` and filters by ID instead of calling the per-model endpoint (which llama.cpp does not support).
- `src/ollama_openai_proxy/translators/response.py` — `models_list_to_tags` populates `details.capabilities`, `size`, `details.parameter_size`, and `details.quantization_level` from upstream architecture and meta fields.

## [0.1.2] - 2026-07-07

- **why:** Fix four streaming defects found during blueprint audit
- **model:** llama-cpp/qwen-3.6-think-coding
- **tags:** streaming, fix, blueprint-audit

### Fixed

- `src/ollama_openai_proxy/translators/streaming_adapter.py` — final `done:true` chunk for completions now yields empty `response` field per Ollama convention (client already has content from intermediate chunks).
- `src/ollama_openai_proxy/translators/streaming.py` — extracts `created` timestamp from the first SSE data line when not provided, so `created_at` is populated in every streaming line.
- `src/ollama_openai_proxy/translators/streaming_adapter.py` — tool call `arguments` are now parsed from JSON strings to dicts in streaming mode, matching non-streaming behavior.
- `src/ollama_openai_proxy/translators/streaming.py` — on stream interruption, accumulated content is emitted before the error line so no generated text is lost.

### Added

- `src/ollama_openai_proxy/translators/streaming_adapter.py` — extracted SSE state machine and chunk builder into a separate module (194 lines) to keep `streaming.py` under 200 lines.
- `tests/test_streaming_gaps.py` — 13 regression tests covering the four streaming gaps identified in the blueprint audit.

### Changed

- `CODEBASE.md` — added `streaming_adapter.py` and `test_streaming_gaps.py` to directory tree and tracing table.

## [0.1.1] - 2026-07-07

- **why:** Add tests to reach 80%+ coverage threshold and create ADR for coverage policy
- **model:** llama-cpp/qwen-3.6-think-coding
- **tags:** test, coverage, adr, quality-gate

### Added

- `tests/test_client.py` — 15 tests for `UpstreamClient` (get, post, stream_post, close, and all error paths).
- `tests/test_request_translator_edge_cases.py` — 19 tests for request translation edge cases (suffix, system as list, images, tool_name absorption, tool_calls passthrough, embed dimensions).
- `tests/test_streaming_edge_cases.py` — 22 tests for streaming adapter edge cases and full async generator integration.
- `tests/test_dependencies.py` — 3 tests for dependency injection (get/set upstream client).
- `docs/adr/adr-001-test-coverage.md` — records 80% coverage threshold, scope exclusions, and enforcement rules.
- `pytest-cov>=7.1.0` — dev dependency for coverage measurement.

### Fixed

- `src/ollama_openai_proxy/client.py` — `stream_post` non-JSON error path no longer crashes on `str.get()` (line 121).

### Changed

- `CODEBASE.md` — added 5 new test file paths to directory tree.

## [0.1.0] - 2026-07-07

- **why:** Implement the full Ollama-to-OpenAI API translation proxy with 15 endpoints, streaming support, and test coverage.
- **model:** llama-cpp/qwen-3.6-think-coding
- **tags:** implementation, proxy, ollama, openai, fastapi, streaming

### Added

- `src/ollama_openai_proxy/` — Python package with CLI, server, router, client, error handling, and configuration modules.
- `src/ollama_openai_proxy/handlers/` — Per-endpoint handlers for `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/ps`, and `/api/version`.
- `src/ollama_openai_proxy/translators/` — Request translators (Ollama to OpenAI), response translators (OpenAI to Ollama), and SSE-to-NDJSON streaming adapter.
- `tests/` — 66 tests covering all handlers, translators, streaming adapter, error handling, and unsupported endpoints.
- `scripts/verify_codebase_sync.sh` — validates that all paths listed in `CODEBASE.md` exist on disk.

### Changed

- `CODEBASE.md` — updated directory tree, dependency manifest (replaced deprecated `tool.uv.dev-dependencies` with `dependency-groups.dev`), and added `dependencies.py` module.

## [0.0.0] - 2026-07-07

- **why:** Initial project setup with architecture specification, documentation, and tooling. No implementation code.
- **model:** opencode/deepseek-v4-pro
- **tags:** architecture, blueprint, documentation, setup

### Added

- `BLUEPRINT.md` — language-agnostic architecture specification with 15 endpoint contracts, field-level translation tables, concrete examples, SSE state machine, and error boundaries.
- `CODEBASE.md` — physical file-to-component mapping with annotated directory tree, dependency manifest, and language-specific implementation notes.
- `README.md` — user-facing documentation covering install methods (uvx, uv tool install, source), quick start, configuration, and supported endpoints.
- `.env.example` — environment variable template for `UPSTREAM_URL`, `LISTEN_HOST`, `LISTEN_PORT`, `REQUEST_TIMEOUT`.
- `.gitignore` — workspace hygiene rules for Python artifacts, virtual environments, and IDE files.
- `VERSION` — semantic version file at `0.0.0`.
- `scripts/bump-version.sh` — version bumping tool supporting major, minor, and patch increments.
- `scripts/validate-changelog.sh` — changelog format validator.
- `docs/memory/` — project memory system with index and initial architecture decision entry.
