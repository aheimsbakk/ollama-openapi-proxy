# ADR-001: Test Coverage Threshold and Scope

- **Status:** Accepted
- **Date:** 2026-07-07
- **Tags:** testing, coverage, quality-gate

## Context

The project requires a minimum of 80% test coverage across all source files. Coverage must reflect actual code paths, not scaffold or boilerplate.

## Decision

1. **80% threshold** applies to the `src/ollama_openai_proxy/` package.
2. Coverage is measured with `pytest-cov` using `--cov=src/ollama_openai_proxy --cov-report=term-missing`.
3. Coverage tests must exercise real business logic: request/response translation, streaming adaptation, error handling, and handler pipelines.
4. Boilerplate files (`__init__.py`, `__main__.py`) are excluded from the threshold. They contain no logic to test.
5. CLI (`cli.py`) and configuration (`config.py`) are excluded — they are thin wrappers around environment variables and argument parsing.
6. The threshold applies to the remaining core modules: `client.py`, `dependencies.py`, `errors.py`, `handlers/*.py`, `router.py`, `server.py`, `translators/*.py`.

## Consequences

- Tests must cover actual translation and proxy logic.
- Scaffold files do not count toward or against the threshold.
- Coverage gaps in core modules are treated as blockers for merge.
- `pytest-cov` is a dev dependency. Run with `uv run pytest --cov=src/ollama_openai_proxy --cov-report=term-missing`.
- The threshold is enforced by the CI check (when CI is added).

## Status

Achieved 89% overall coverage (132 tests) as of 2026-07-07. Excluding scaffold files: 90% coverage across core modules.
