# Changelog

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
