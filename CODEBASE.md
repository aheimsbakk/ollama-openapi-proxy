# CODEBASE.md — Physical File Mapping

## Language & Framework Specification

| Concern | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Specified in `tasks/project.md`. |
| **HTTP framework** | FastAPI | Native async support, built-in SSE streaming via `StreamingResponse`, automatic request validation via Pydantic models. The async foundation maps cleanly to the proxy's need to fan out streaming responses without blocking. |
| **ASGI server** | uvicorn | Standard production ASGI server for FastAPI. |
| **HTTP client** | httpx | Async HTTP client with connection pooling, timeout control, and native SSE iteration. Required for streaming upstream requests without blocking the event loop. |
| **Package manager** | uv | Specified in `tasks/project.md`. Manages virtual environment, dependency resolution, and package installation. |
| **CLI argument parsing** | argparse (stdlib) | Specified in `tasks/project.md`: "use builtin for argument parsing". No external CLI framework. |
| **Naming convention** | `kebab-case` for non-Python files; `snake_case` for Python modules per community convention | Python imports require valid identifiers; hyphens are illegal. |

---

## Dependency Manifest

```toml
[project]
name = "ollama-openai-proxy"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
    "httpx>=0.27,<1",
]

[project.scripts]
ollama-openai-proxy = "ollama_openai_proxy.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=9,<10",
    "pytest-asyncio>=1,<2",
    "httpx>=0.27,<1",          # also used in tests for TestClient
]
```

All Pydantic models used for request/response validation are defined inline within the handler modules. No separate `models.py` file — the schemas are small and closely coupled to their single handler.

---

## Directory Tree & Blueprint Mapping

```
work/                                   # Repository root
├── BLUEPRINT.md                        # Architecture specification
├── CODEBASE.md                         # This file
├── README.md                           # User-facing documentation
├── pyproject.toml                      # Project metadata, deps, entry points
├── .gitignore                          # Workspace hygiene
├── .env.example                        # Environment variable template
│
├── src/
│   └── ollama_openai_proxy/           # Python package (snake_case: Python import requirement)
│       │
│       ├── __init__.py                # Package marker, version export
│       │
│       ├── __main__.py                # Allows `python -m ollama_openai_proxy`
│       │   Maps to: (no Blueprint component — thin launcher)
│       │
│       ├── cli.py                     # argparse setup, entry point: main()
│       │   Maps to: External Configuration (Blueprint §6)
│       │   Parses: --host/-H, --port/-p, --upstream-url/-u, --timeout/-t,
│       │   --version/-V, --help/-h, --verbosity/-v (count: -v/-vv/-vvv)
│       │   Falls back to env vars: LISTEN_HOST, LISTEN_PORT, UPSTREAM_URL,
│       │   REQUEST_TIMEOUT. Verbosity sets logging level (default: ERROR).
│       │
│       ├── config.py                  # Config dataclass loaded from env/args
│       │   Maps to: External Configuration (Blueprint §6)
│       │   Merges CLI args with env vars; env vars take precedence over
│       │   defaults, CLI args take precedence over env vars.
│       │
│       ├── dependencies.py            # FastAPI dependency injection for upstream client
│       │   Maps to: HTTP Client (Blueprint §2)
│       │   Provides: get_upstream_client() dependency, set_upstream_client() setter.
│       │   Breaks circular import between server.py and handlers/*.py.
│       │   Maps to: External Configuration (Blueprint §6)
│       │   Merges CLI args with env vars; env vars take precedence over
│       │   defaults, CLI args take precedence over env vars.
│       │
│       ├── server.py                  # FastAPI app creation, uvicorn.run()
│       │   Maps to: HTTP Server Layer (Blueprint §2)
│       │   Creates the FastAPI instance. Registers all route handlers.
│       │   Configures CORS (allow all origins for drop-in compatibility).
│       │   Installs global exception handler for uncaught errors → 502/504.
│       │
│       ├── router.py                  # Route registration, path → handler mapping
│       │   Maps to: Router (Blueprint §2)
│       │   Registers all Ollama API paths with their handler functions.
│       │   ~15 route registrations in a single `register_routes(app)` call.
│       │
│       ├── client.py                  # Async HTTP client for upstream calls
│       │   Maps to: HTTP Client (Blueprint §2)
│       │   Provides: post(url, json), get(url), stream_post(url, json)
│       │   All calls use httpx.AsyncClient with configured timeout.
│       │   Connection errors caught → AppError with status 502.
│       │   Timeout errors caught → AppError with status 504.
│       │
│       ├── errors.py                  # Error response builders
│       │   Maps to: Error Boundaries (Blueprint §7)
│       │   AppError exception class (message + status_code).
│       │   build_error_response(exc) → FastAPI JSONResponse.
│       │   Error message format: {"error": "<plain language message>"}
│       │
│       ├── handlers/                  # Per-endpoint handler modules
│       │   │
│       │   ├── __init__.py
│       │   │
│       │   ├── generate.py            # POST /api/generate
│       │   │   Maps to: §3.1 POST /api/generate
│       │   │   async handler. Reads Ollama body → translates via
│       │   │   translators.request.generate_to_completion() →
│       │   │   calls client.stream_post() or client.post() →
│       │   │   translates response via translators.response.completion_to_generate().
│       │   │
│       │   ├── chat.py                # POST /api/chat
│       │   │   Maps to: §3.1 POST /api/chat
│       │   │   async handler. Same pipeline pattern as generate.
│       │   │   Uses chat_to_chat_completions() / chat_completion_to_chat().
│       │   │
│       │   ├── embed.py               # POST /api/embed + POST /api/embeddings
│       │   │   Maps to: §3.1 POST /api/embed, POST /api/embeddings
│       │   │   Two async handlers in one file (shared translation logic).
│       │   │   embed_to_embeddings() / embeddings_to_embed().
│       │   │   Legacy handler: embeddings_legacy_to_embeddings().
│       │   │
│       │   └── models.py              # GET /api/tags, POST /api/show, GET /api/ps
│       │       Maps to: §3.2 Informational Endpoints
│       │       Three sync handlers (upstream GET calls).
│       │       models_list_to_tags() / models_show_to_show() / models_list_to_ps().
│       │       Also: version() handler for GET /api/version (static).
│       │
│       └── translators/               # Request/response translation logic
│           │
│           ├── __init__.py
│           │
│           ├── request.py             # Ollama → OpenAI request translators
│           │   Maps to: Request Translators (Blueprint §2, §3)
│           │   Pure functions. No I/O. No framework imports.
│           │   Functions per endpoint pair:
│           │     generate_to_completion(ollama_body) → openai_body
│           │     chat_to_chat_completions(ollama_body) → openai_body
│           │     embed_to_embeddings(ollama_body) → openai_body
│           │   Handles: images → content parts, format → response_format,
│           │   options → flat params, field drops.
│           │
│           ├── response.py            # OpenAI → Ollama response translators
│           │   Maps to: Response Translators (Blueprint §2, §3)
│           │   Pure functions. No I/O.
│           │   Functions per endpoint pair:
│           │     completion_to_generate(openai_body) → ollama_body
│           │     chat_completion_to_chat(openai_body) → ollama_body
│           │     embeddings_to_embed(openai_body) → ollama_body
│           │     models_list_to_tags(openai_body) → ollama_body
│           │     model_show_to_show(openai_body) → ollama_body
│           │   Handles: timestamp conversion, field renaming, structure reshaping.
│           │
│           ├── streaming.py           # SSE-to-NDJSON stream adapter
│           │   Maps to: Stream Adapter (Blueprint §2, §9)
│           │   Async generator: sse_to_ollama_stream(httpx_response, translator_fn)
│           │   Reads SSE lines from the upstream response.
│           │   Parses `data:` lines, passes JSON chunks through translator_fn,
│           │   yields serialized Ollama NDJSON lines.
│           │   Extracts created timestamp from first chunk if not provided.
│           │
│           ├── streaming_adapter.py   # SSE state machine and chunk builder
│           │   Maps to: Stream Adapter (Blueprint §2, §9)
│           │   SSEAdapter class: processes SSE lines, accumulates state,
│           │   builds Ollama-format response objects. Handles tool call
│           │   argument parsing (string → dict) and final-chunk empty
│           │   response per Ollama convention.
│           │
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures: mock upstream server,
│   │                                   # FastAPI TestClient, sample payloads
│   ├── test_generate.py               # Unit tests for /api/generate pipeline
│   ├── test_chat.py                   # Unit tests for /api/chat pipeline
│   ├── test_embed.py                  # Unit tests for /api/embed pipeline
│   ├── test_models.py                 # Unit tests for /api/tags, /api/show, /api/ps
│   ├── test_errors.py                # Error handling & boundary tests
│   ├── test_client.py                # UpstreamClient HTTP client tests
│   ├── test_dependencies.py          # Dependency injection tests
│   ├── test_translators.py           # Pure-function tests for translators
│   ├── test_request_translator_edge_cases.py  # Request translation edge cases
│   ├── test_streaming.py             # Streaming adapter unit tests
│   ├── test_streaming_edge_cases.py  # Streaming adapter integration & edge cases
│   └── test_streaming_gaps.py        # Regression tests for blueprint streaming gaps
│
└── scripts/
    └── verify_codebase_sync.sh        # Validates CODEBASE.md paths exist
```

---

## Entry Points

| Entry Point | Path | How to Invoke |
|---|---|---|
| CLI (primary) | `src/ollama_openai_proxy/cli.py::main()` | `ollama-openai-proxy --host 0.0.0.0 --port 11434` |
| Module runner | `src/ollama_openai_proxy/__main__.py` | `python -m ollama_openai_proxy` |
| FastAPI app | `src/ollama_openai_proxy/server.py::create_app()` | Imported by `cli.py` and test `conftest.py` |

---

## Tracing: Blueprint Component → File

| Blueprint Component (§) | Physical File |
|---|---|
| HTTP Server Layer (§2) | `src/ollama_openai_proxy/server.py` |
| Router (§2) | `src/ollama_openai_proxy/router.py` |
| Handler Pipeline (§2) | `src/ollama_openai_proxy/handlers/*.py` + `translators/*.py` |
| Request Translators (§3) | `src/ollama_openai_proxy/translators/request.py` |
| Response Translators (§3) | `src/ollama_openai_proxy/translators/response.py` |
| HTTP Client (§2) | `src/ollama_openai_proxy/client.py` |
| Stream Adapter (§2, §9) | `src/ollama_openai_proxy/translators/streaming.py` + `translators/streaming_adapter.py` |
| Error Boundaries (§7) | `src/ollama_openai_proxy/errors.py` |
| External Configuration (§6) | `src/ollama_openai_proxy/config.py` + `cli.py` |
| Unsupported Endpoints (§3.3) | `src/ollama_openai_proxy/router.py` (501 return in route registration) |

---

## Language-Specific Implementation Notes

### Async foundation

FastAPI handlers are `async def`. The entire translation pipeline — receiving the Ollama request, calling upstream via `httpx`, streaming the response — runs on the asyncio event loop. This lets the proxy handle multiple concurrent connections without thread pools.

### Streaming via async generator

FastAPI's `StreamingResponse` accepts an async generator. The `streaming.py` module implements the SSE → NDJSON conversion as an async generator:

```
async def sse_to_ollama_stream(upstream_response, chunk_translator) → AsyncIterator[str]
```

Each yielded string is one Ollama NDJSON line. FastAPI flushes each chunk immediately (no buffering) because the generator yields one line at a time.

### httpx timeout configuration

`httpx.AsyncClient` is created once at startup (single instance for connection pooling) with `timeout=httpx.Timeout(connect=10, read=REQUEST_TIMEOUT, write=10, pool=10)`. The read timeout equals the configured `REQUEST_TIMEOUT` env var.

### Pydantic models: local, not global

Request/response validation Pydantic models are defined in each handler file, not in a shared `models.py`. Rationale: each Ollama endpoint has its own schema, and no schema is shared across endpoints. Grouping them would create an import tangle. Each handler file owns the schema it validates.

### Image content mapping

Ollama's `images` field (list of base64 strings) maps to OpenAI's content-parts array. The translator wraps each image as:
```
{"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}
```
The original `content` string becomes a `{"type": "text", "text": "<content>"}` part.

### options → flat params

Ollama's nested `options` object is flattened by the translator. Example: `{"options": {"temperature": 0.7, "top_p": 0.9}}` becomes `{"temperature": 0.7, "top_p": 0.9}` in the OpenAI request body. Unknown option keys are silently dropped. Mapped keys are listed in BLUEPRINT.md §3.1.

### timestamp conversion

OpenAI uses Unix timestamps (integer seconds). Ollama uses ISO 8601 strings. The translator function `_ts_to_iso8601(unix_ts: int) → str` handles this conversion. Returns `""` if `created` is absent from the upstream response.
