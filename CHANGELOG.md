# Changelog

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
