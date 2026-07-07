# Ollama-to-OpenAI Proxy

> [github.com/aheimsbakk/ollama-openai-proxy](https://github.com/aheimsbakk/ollama-openai-proxy)

An API translation proxy that speaks the Ollama API on the front and forwards
requests to an OpenAI-compatible backend.

## What it does

- Accepts requests in the Ollama API format (`/api/generate`, `/api/chat`,
  `/api/embed`) from any Ollama-compatible tool or client.
- Translates them to OpenAI-compatible API calls and sends them to an upstream
  server such as `llamacpp-server`.
- Translates responses back to the Ollama format, including streaming.
- Lets you use Ollama-only tools with any OpenAI-compatible server — no
  configuration changes needed on the client side.

## Install

### Run without installing (uvx)

Run the proxy directly from the repository in a temporary environment:

```shell
uvx --from git+https://github.com/aheimsbakk/ollama-openai-proxy \
  ollama-openai-proxy --upstream-url http://localhost:8080/v1
```

### Run a specific version

Use a Git tag to pin to a release:

```shell
uvx --from git+https://github.com/aheimsbakk/ollama-openai-proxy@v0.1.0 \
  ollama-openai-proxy --upstream-url http://localhost:8080/v1
```

### Install permanently

Install the proxy as a system-wide tool:

```shell
uv tool install git+https://github.com/aheimsbakk/ollama-openai-proxy
```

After installing, run it directly:

```shell
ollama-openai-proxy --upstream-url http://localhost:8080/v1
```

### Install a specific version

```shell
uv tool install git+https://github.com/aheimsbakk/ollama-openai-proxy@v0.1.0
```

### From source (for development)

```shell
git clone https://github.com/aheimsbakk/ollama-openai-proxy
cd ollama-openai-proxy
uv sync

uv run ollama-openai-proxy --upstream-url http://localhost:8080/v1
```

## Quick start

Start your upstream server (for example, `llamacpp-server` on port 8080).
Then start the proxy with one of the install methods above.

The proxy listens on `http://localhost:11434` by default — the same port
Ollama uses.

Point any Ollama-compatible tool at `http://localhost:11434`. It will
work without changes.

```shell
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Usage

### With command-line arguments

```shell
ollama-openai-proxy \
  --host 0.0.0.0 \
  --port 11434 \
  --upstream-url http://llamacpp:8080/v1 \
  --timeout 300
```

### With environment variables

```shell
export UPSTREAM_URL=http://llamacpp:8080/v1
export LISTEN_HOST=0.0.0.0
export LISTEN_PORT=11434
export REQUEST_TIMEOUT=300

ollama-openai-proxy
```

### From source checkout

```shell
uv run ollama-openai-proxy --upstream-url http://localhost:8080/v1
```

### As a Python module (development use)

```shell
python -m ollama_openai_proxy --upstream-url http://localhost:8080/v1
```

## Configuration

All settings accept both a command-line flag and an environment variable.
Command-line flags take precedence over environment variables.

### Runtime settings

| Long | Short | Env variable | Default | Description |
|---|---|---|---|---|
| `--host` | `-H` | `LISTEN_HOST` | `0.0.0.0` | Network interface to bind |
| `--port` | `-p` | `LISTEN_PORT` | `11434` | TCP port to listen on |
| `--upstream-url` | `-u` | `UPSTREAM_URL` | `http://localhost:8080/v1` | Base URL of the upstream OpenAI-compatible server (include `/v1` path) |
| `--timeout` | `-t` | `REQUEST_TIMEOUT` | `300` | Maximum seconds to wait for an upstream response |

### Informational flags

| Long | Short | Description |
|---|---|---|
| `--help` | `-h` | Print usage information and exit |
| `--version` | `-V` | Print the program version and exit |
| `--verbosity` | `-v` | Increase log verbosity. Repeat: `-v` for WARNING, `-vv` for INFO, `-vvv` for DEBUG. Default: ERROR |

## Supported endpoints

| Ollama endpoint | Status | Notes |
|---|---|---|
| `POST /api/generate` | Supported | Translated to `/v1/completions` |
| `POST /api/chat` | Supported | Translated to `/v1/chat/completions` |
| `POST /api/embed` | Supported | Translated to `/v1/embeddings` |
| `POST /api/embeddings` | Supported (legacy) | Translated to `/v1/embeddings` |
| `GET /api/tags` | Supported | Translated to `GET /v1/models` |
| `POST /api/show` | Supported | Translated to `GET /v1/models/{model}` |
| `GET /api/ps` | Supported | Translated to `GET /v1/models` |
| `GET /api/version` | Supported | Returns static version `0.0.0-proxy` |
| `POST /api/create` | Not supported | Returns HTTP 501 |
| `POST /api/copy` | Not supported | Returns HTTP 501 |
| `DELETE /api/delete` | Not supported | Returns HTTP 501 |
| `POST /api/pull` | Not supported | Returns HTTP 501 |
| `POST /api/push` | Not supported | Returns HTTP 501 |
| Blob endpoints | Not supported | Returns HTTP 501 |

Model management endpoints (create, copy, delete, pull, push) are not
supported because the upstream OpenAI-compatible server manages its own
model lifecycle. The proxy has no model storage.

## Developing

### Setup

```shell
uv sync          # Install dependencies
uv run pytest    # Run the test suite
```

### Project structure

```
src/ollama_openai_proxy/
├── cli.py          # CLI entry point, argument parsing
├── server.py       # FastAPI app creation
├── router.py       # Route registration
├── client.py       # Async HTTP client for upstream calls
├── config.py       # Configuration loading
├── errors.py       # Error response builders
├── handlers/       # Per-endpoint HTTP handlers
│   ├── generate.py # POST /api/generate
│   ├── chat.py     # POST /api/chat
│   ├── embed.py    # POST /api/embed + /api/embeddings
│   └── models.py   # GET /api/tags, /api/show, /api/ps, /api/version
└── translators/    # Request/response translation logic
    ├── request.py  # Ollama → OpenAI
    ├── response.py # OpenAI → Ollama
    ├── streaming.py # SSE-to-NDJSON stream adapter
    └── streaming_adapter.py # SSE state machine and chunk builder
```

See `BLUEPRINT.md` for the architecture specification and `CODEBASE.md` for
detailed file-to-component mapping.

## Contributing

Report issues or suggest improvements through the project's issue tracker.
Pull requests should include tests for new functionality.
