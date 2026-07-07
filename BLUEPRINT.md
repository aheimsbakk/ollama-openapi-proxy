# BLUEPRINT.md — Language-Agnostic Architecture Specification

## 1. System Goals

Build a stateless API translation middleware that:

1. **Exposes an Ollama-compatible HTTP API** — drop-in replacement for Ollama server endpoints.
2. **Translates incoming Ollama requests** to equivalent OpenAI-compatible API calls.
3. **Forwards translated requests** to a configurable upstream OpenAI-compatible server (primary target: `llamacpp-server`).
4. **Translates upstream OpenAI responses** back to Ollama-compatible response structures.
5. **Supports streaming end-to-end** — SSE from upstream converted to Ollama's newline-delimited JSON stream format.
6. **Operates with minimal configuration** — a single environment variable for the upstream URL.

The primary use case: tools that only speak the Ollama API can drive a `llamacpp-server` instance without modification.

---

## 2. Component Hierarchy

```
┌─────────────────────────────────────────────────────┐
│ HTTP Server Layer                                   │
│ Binds to LISTEN_HOST:LISTEN_PORT. Accepts JSON and  │
│ streaming requests from Ollama-compatible clients.  │
└───────────┬─────────────────────────────────────────┘
            │ dispatches by path
┌───────────▼─────────────────────────────────────────┐
│ Router                                              │
│ Maps each Ollama API path to a handler. Supports    │
│ exact and prefix matching for parameterized routes. │
│ Rejects unknown paths with 404.                     │
└───────────┬─────────────────────────────────────────┘
            │ per-endpoint delegation
┌───────────▼─────────────────────────────────────────┐
│ Handler Pipeline (per endpoint)                     │
│ ┌──────────────────┐  ┌───────────────────┐         │
│ │ Request          │  │ HTTP Client       │         │
│ │ Translator       │──│ (async, timeout)  │─────────┼──► Upstream
│ │ Ollama→OpenAI    │  │                   │         │
│ └──────────────────┘  └───────────────────┘         │
│ ┌──────────────────┐  ┌───────────────────┐         │
│ │ Response         │  │ Stream Adapter    │         │
│ │ Translator       │  │ SSE→Ollama NDJSON │         │
│ │ OpenAI→Ollama    │  │ (for stream=true) │         │
│ └──────────────────┘  └───────────────────┘         │
└─────────────────────────────────────────────────────┘
```

**Handler pipeline flow** (non-streaming):

1. Request Translator reads the Ollama request body, produces an OpenAI request body.
2. HTTP Client sends the OpenAI request to upstream, returns the OpenAI response body.
3. Response Translator reads the OpenAI response body, produces an Ollama response body.
4. HTTP Server returns the Ollama response to the client.

**Handler pipeline flow** (streaming):

1. Request Translator reads the Ollama request body, produces an OpenAI request body with `stream: true`.
2. HTTP Client opens a streaming connection to upstream, receives an SSE stream.
3. Stream Adapter parses SSE events into individual JSON chunks.
4. Response Translator transforms each chunk into an Ollama-format JSON line, flushed immediately.
5. When the upstream stream ends, a final `done: true` response line is emitted with accumulated usage statistics.

---

## 3. Endpoint Contracts

### 3.0 Ollama API Surface Summary

The proxy exposes these Ollama API endpoints. Each maps to an OpenAI equivalent
or returns a static/stub response.

| Method | Path | Required fields | Mapped to | Proxy support |
|---|---|---|---|---|
| `POST` | `/api/generate` | `model` | `POST /v1/completions` | Full translation |
| `POST` | `/api/chat` | `model`, `messages` | `POST /v1/chat/completions` | Full translation |
| `POST` | `/api/embed` | `model`, `input` | `POST /v1/embeddings` | Full translation |
| `POST` | `/api/embeddings` | `model`, `prompt` | `POST /v1/embeddings` | Full translation (legacy) |
| `GET` | `/api/tags` | none | `GET /v1/models` | Full translation |
| `POST` | `/api/show` | `model` | `GET /v1/models/{model}` | Full translation |
| `GET` | `/api/ps` | none | `GET /v1/models` | Full translation |
| `GET` | `/api/version` | none | (static) | Returns `{"version":"0.0.0-proxy"}` |
| `POST` | `/api/create` | `model` | — | Returns 501 |
| `POST` | `/api/copy` | `source`, `destination` | — | Returns 501 |
| `DELETE` | `/api/delete` | `model` | — | Returns 501 |
| `POST` | `/api/pull` | `model` | — | Returns 501 |
| `POST` | `/api/push` | `model` | — | Returns 501 |
| `HEAD` | `/api/blobs/:digest` | — | — | Returns 501 |
| `POST` | `/api/blobs/:digest` | — | — | Returns 501 |

**Ollama response conventions (all endpoints):**
- The top-level `model` field echoes the model name from the request.
- The `done` field is `false` for streaming chunks, `true` for the final object.
- Timing/duration fields (`total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`) are in nanoseconds.
- Token counts (`prompt_eval_count`, `eval_count`) are integers.
- `created_at` is an ISO 8601 timestamp string.

**Ollama streaming format:**
- Each response line is a single, complete JSON object terminated by a newline (`\n`).
- No SSE wrapper. No `data:` prefix. No `\n\n` terminator between objects.
- The client reads one line at a time and parses each as JSON.
- The final object always has `"done": true`.

**Ollama error format:**
- Error responses are JSON objects with a single `error` field:
  ```json
  {"error": "plain-language description of the problem"}
  ```
- HTTP status codes: 400 (bad request), 404 (not found), 500 (server error).

---

### 3.1 Primary Inference Endpoints

#### POST /api/generate → POST /v1/completions

Converts an Ollama completion request to an OpenAI legacy completions request.

**Request Translation Rules (Ollama → OpenAI):**

| Ollama field | OpenAI field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `prompt` | `prompt` | Direct copy. |
| `suffix` | `suffix` | Direct copy. |
| `images` | `messages[].content` | Prepend a system message with image content parts (type `image_url`, base64 data URI). |
| `system` | Prepend to prompt or add as system message | If the upstream model supports system prompts, prepend bounded instructions to the prompt string. |
| `format` | `response_format` | `"json"` → `{"type":"json_object"}`. A JSON schema object → `{"type":"json_schema","json_schema":{...}}`. |
| `options.temperature` | `temperature` | Direct copy. |
| `options.top_p` | `top_p` | Direct copy. |
| `options.top_k` | Dropped | Not supported by the OpenAI completions schema. |
| `options.stop` | `stop` | Direct copy (string or array of strings). |
| `options.seed` | `seed` | Direct copy. |
| `options.num_predict` | `max_tokens` | Direct copy. |
| `options.frequency_penalty` | `frequency_penalty` | Direct copy. |
| `options.presence_penalty` | `presence_penalty` | Direct copy. |
| `options.num_ctx` | Dropped | Context size is managed by the upstream server. |
| `stream` | `stream` | Direct copy. |
| `raw` | Dropped | No OpenAI equivalent. |
| `keep_alive` | Dropped | Middleware is stateless; model lifecycle is managed by the upstream server. |
| `context` | Dropped | Middleware is stateless. Clients managing context should serialize it into the prompt. |
| `template` | Dropped | Template is upstream responsibility. |

**Response Translation Rules (OpenAI → Ollama):**

| OpenAI field | Ollama field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `created` | `created_at` | Convert Unix timestamp integer to ISO 8601 string. |
| `choices[0].text` | `response` | Direct copy. |
| `choices[0].finish_reason` | `done_reason` | Map `stop`/`length`/`content_filter` directly. |
| `usage.prompt_tokens` | `prompt_eval_count` | Direct copy. |
| `usage.completion_tokens` | `eval_count` | Direct copy. |
| `usage` timing fields | `total_duration`, `prompt_eval_duration`, `eval_duration` | Derived fields; set to `0` or approximate from upstream if available. |
| — | `done` | Derived: `false` during stream, `true` for final or non-streamed response. |

**Streaming:** Each OpenAI SSE chunk becomes one Ollama NDJSON line. The `response` field in streaming chunks contains the delta text from `choices[0].text`. The final response object includes the accumulated usage statistics.

---

#### POST /api/chat → POST /v1/chat/completions

Converts an Ollama chat request to an OpenAI chat completions request.

**Request Translation Rules (Ollama → OpenAI):**

| Ollama field | OpenAI field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `messages` | `messages` | Direct structural copy. Each message object maps `role`→`role`, `content`→`content`. |
| `messages[].images` | `messages[].content` | Convert to array of content parts: text part from existing `content` + image part per base64 entry. |
| `messages[].tool_calls` | `messages[].tool_calls` | Direct copy. |
| `messages[].tool_name` | (absorb into content) | Prepend tool result context to the `tool` role message content. |
| `messages[].thinking` | Dropped | Upstream-specific; not passed to OpenAI. |
| `tools` | `tools` | Direct copy. |
| `format` | `response_format` | Same rules as `/api/generate`. |
| `options.*` | (same mapping as `/api/generate`) | Options mapping is identical. |
| `stream` | `stream` | Direct copy. |
| `keep_alive` | Dropped | Stateless middleware. |

**Response Translation Rules (OpenAI → Ollama):**

| OpenAI field | Ollama field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `created` | `created_at` | Convert Unix timestamp to ISO 8601. |
| `choices[0].message.role` | `message.role` | Direct copy. |
| `choices[0].message.content` | `message.content` | Direct copy. |
| `choices[0].message.tool_calls` | `message.tool_calls` | Direct copy. |
| `choices[0].finish_reason` | `done_reason` | Direct copy. |
| `usage.prompt_tokens` | `prompt_eval_count` | Direct copy. |
| `usage.completion_tokens` | `eval_count` | Direct copy. |
| — | `done` | Derived from stream state or non-stream final. |

**Streaming:** Each OpenAI SSE chunk produces one Ollama NDJSON line. The `message.content` field in streaming chunks contains the delta. Tool call deltas are accumulated and emitted when complete.

---

#### POST /api/embed → POST /v1/embeddings

Converts an Ollama embedding request to an OpenAI embeddings request.

| Ollama field | OpenAI field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `input` | `input` | Direct copy (string or array of strings). |
| `truncate` | (applied client-side) | If `false` and input exceeds context, return an error before forwarding. |
| `options` | `dimensions` | If `options` contains dimension hints, map to `dimensions`. Other options dropped. |
| `keep_alive` | Dropped | Stateless. |

**Response Translation:**

| OpenAI field | Ollama field | Notes |
|---|---|---|
| `model` | `model` | Direct copy. |
| `data[].embedding` | `embeddings` | Array of embedding arrays. |
| `usage.prompt_tokens` | `prompt_eval_count` | Direct copy. |
| — | `total_duration`, `load_duration` | Set to `0`. |

---

#### POST /api/embeddings (legacy) → POST /v1/embeddings

Accepts `prompt` (single string). Maps `prompt` → `input`. Response maps `data[0].embedding` → `embedding` (flat list). Other fields identical to `/api/embed`.

---

### 3.2 Informational Endpoints

#### GET /api/tags → GET /v1/models

Fetches the upstream model list. Transforms `data[]` (each with `id`, `created`, `owned_by`) into Ollama `models[]` format:

| Ollama output field | Source |
|---|---|
| `name` | `id` |
| `model` | `id` |
| `modified_at` | Convert `created` Unix timestamp to ISO 8601. |
| `size` | `0` (unavailable from OpenAI models endpoint). |
| `digest` | Hash of `id` for uniqueness. |
| `details.format` | `"gguf"` (assumed). |
| `details.family` | Derived from model name prefix. |
| `details.parameter_size` | `""` (unknown). |
| `details.quantization_level` | `""` (unknown). |

---

#### POST /api/show → GET /v1/models/{model}

Extracts the `model` field from the request body. Fetches the single model from upstream. Returns Ollama-format model info. Fields not available from OpenAI (`modelfile`, `template`, `parameters`, `model_info`) are returned as empty strings or null. The `details` block is populated from the models endpoint data.

---

#### GET /api/ps

Returns the upstream model list in simplified Ollama format. Equivalent to `/api/tags` but adds `expires_at` (set to far future) and `size_vram` (set to `0`).

---

#### GET /api/version

Returns a static response:
```
{ "version": "0.0.0-proxy" }
```

---

### 3.3 Model Management Endpoints (Unsupported)

These endpoints are defined by the Ollama API but have no equivalent on the upstream OpenAI server. Each returns HTTP `501 Not Implemented` with a descriptive error message.

- `POST /api/create`
- `POST /api/copy`
- `DELETE /api/delete`
- `POST /api/pull`
- `POST /api/push`
- `HEAD /api/blobs/:digest`
- `POST /api/blobs/:digest`

---

## 4. State Management

The middleware is **entirely stateless**. It holds:

- **No session state**: each request is independent.
- **No model cache**: model loading/unloading is the upstream server's responsibility.
- **No conversation context**: `context` and `keep_alive` fields are ignored.
- **Configuration**: loaded once at process startup from environment variables; treated as immutable.

---

## 5. Persistence

No persistent storage. No database. No filesystem writes. All state exists only for the duration of a single HTTP request/response cycle.

---

## 6. External Configuration

All settings accept both a command-line flag and an environment variable.
Command-line flags take precedence over environment variables.

### Runtime settings

| CLI (long) | CLI (short) | Env variable | Default | Purpose |
|---|---|---|---|---|
| `--host` | `-H` | `LISTEN_HOST` | `0.0.0.0` | Network interface to bind the Ollama-facing server. |
| `--port` | `-p` | `LISTEN_PORT` | `11434` | TCP port to bind the Ollama-facing server. |
| `--upstream-url` | `-u` | `UPSTREAM_URL` | `http://localhost:8080/v1` | Base URL of the OpenAI-compatible upstream server. Must include the `/v1` path prefix. |
| `--timeout` | `-t` | `REQUEST_TIMEOUT` | `300` | Maximum time in seconds to wait for an upstream response. |

### Informational flags (CLI only, no env variable)

| CLI (long) | CLI (short) | Purpose |
|---|---|---|
| `--help` | `-h` | Print usage information and exit. |
| `--version` | `-V` | Print the program version and exit. |
| `--verbosity` | `-v` | Increase log verbosity. Repeatable: `-v` for WARNING, `-vv` for INFO, `-vvv` for DEBUG. Default: ERROR. |

---

## 7. Error Boundaries

### 7.1 Client-Facing Errors (Ollama-Compatible)

| Situation | HTTP Status | Response Format |
|---|---|---|
| Malformed JSON request body | 400 | `{"error": "invalid request body: <details>"}` |
| Missing required field (`model`) | 400 | `{"error": "missing required field: model"}` |
| Unsupported endpoint | 501 | `{"error": "endpoint not supported by this proxy: <path>"}` |
| Unknown endpoint path | 404 | `{"error": "not found"}` |
| Upstream connection failure | 502 | `{"error": "upstream server unreachable: <details>"}` |
| Upstream timeout | 504 | `{"error": "upstream request timed out after N seconds"}` |
| Upstream returns non-200 | (relay) | Proxies the upstream error body when available; falls back to a generic error with the upstream status code. |
| Streaming interruption | (stream terminates) | If the upstream stream breaks mid-response, emit the collected text so far followed by an error line. |

### 7.2 Streaming Error Handling

When an error occurs during a streaming response, the proxy must handle it
differently from a non-streaming error. The HTTP response has already started
(200 status sent, headers flushed), so the error must be communicated
in-stream.

**Upstream returns a non-200 during streaming setup:**
The stream has not started yet. Return the error as a standard non-streaming
error response with the upstream's status code and body (or a 502 if the
upstream body is empty).

**Upstream connection breaks mid-stream:**
The SSE connection is lost before `[DONE]` is received. The proxy must:

1. Emit the Ollama NDJSON line for the last successfully parsed SSE chunk
   (text accumulated so far, `done: false`).
2. Emit a final error object as NDJSON:
   ```json
   {"model": "<model>", "error": "upstream connection lost mid-response", "done": true}
   ```
3. Close the connection.

The `done: true` in the error object signals to the client that the stream
has terminated. The `error` field follows the Ollama error convention.

**Upstream returns a non-200 mid-stream (unusual but possible):**
Some upstream servers send an error chunk in the SSE stream instead of
terminating the connection. If an SSE `data:` line contains an error
structure (contains `"error"` key at the top level), the proxy must:

1. Stop processing further SSE events.
2. Emit a final Ollama NDJSON error object with the upstream's error message.
3. Close the connection.

**Upstream timeout during streaming:**
If no SSE event is received within the configured `REQUEST_TIMEOUT`, the proxy
must:

1. Emit the accumulated response so far as an NDJSON line.
2. Emit a final error object: `{"model": "<model>", "error": "upstream request timed out after N seconds", "done": true}`.
3. Close the connection.

**Malformed SSE line:**
If a `data:` line contains text that is not valid JSON (and is not `[DONE]`),
the proxy must:

1. Skip the malformed line (log a warning).
2. Continue processing the next SSE events.
3. If the stream ends normally after skipping, emit the final `done: true`
   object normally with whatever usage stats were accumulated.

**Accumulated state during streaming:**
Throughout the stream, the proxy accumulates:
- `content`: concatenated text/content from all chunks.
- `tool_calls`: merged tool call deltas from chunks.
- `usage`: the last `usage` object seen (some upstreams send it only in the
  final chunk; some send it incrementally).
- `finish_reason`: the last `finish_reason` seen.

If the stream terminates with an error, the accumulated content is emitted
in the last successful chunk before the error line. This ensures no generated
text is lost.

---

### 7.3 Internal Error Handling

- Every upstream HTTP call must use the configured timeout. Connections that exceed the timeout are aborted, and a 504 is returned.
- If the response translator encounters an unexpected OpenAI response structure, it logs the mismatch and returns a 502 with a description of the schema mismatch.
- The server must not crash on any individual request failure. All exceptions in request handling must be caught at the outermost handler boundary and converted to error responses.

---

## 8. Data Flow Summary

```
Ollama Client
    │
    │ POST /api/chat {"model":"llama3.2","messages":[...]}
    ▼
┌──────────────────────────────────────────────────────┐
│ Ollama-to-OpenAI Proxy (this program)                │
│                                                      │
│  1. Parse Ollama request body                        │
│  2. Route to /api/chat handler                       │
│  3. Translate to OpenAI body:                        │
│     {"model":"llama3.2","messages":[...],"stream":T} │
│  4. POST to UPSTREAM_URL/chat/completions            │
│  5. Receive SSE stream                               │
│  6. For each SSE chunk, translate to Ollama NDJSON:  │
│     {"model":"...","message":{"role":"assistant",    │
│      "content":"The"},"done":false}                  │
│  7. On stream end, emit final done:true line         │
└──────────────────────────────────────────────────────┘
    │
    │ NDJSON stream (Ollama format)
    ▼
Ollama Client
```

---

## 9. Streaming Protocol

**Ollama streaming format:** Each response line is a complete JSON object terminated by a newline (`\n`). The stream continues with `"done": false` objects until the final object with `"done": true`.

**OpenAI streaming format (SSE):** Lines prefixed with `data: ` containing JSON objects, terminated by `\n\n`. The stream ends with `data: [DONE]`.

**Adapter behavior:**
1. Read SSE events from the upstream connection.
2. For each `data:` line that is not `[DONE]`, parse the JSON object.
3. Transform the OpenAI chunk to the corresponding Ollama chunk.
4. Serialize as a single JSON line with a trailing newline.
5. Flush immediately (no buffering).
6. Accumulate usage statistics as they arrive.
7. When `[DONE]` is received, emit the final Ollama response object with `"done": true` and the accumulated usage stats.
8. Close the connection.

---

## Appendix A: Concrete Translation Examples

Each example shows the complete chain: Ollama request → OpenAI request →
OpenAI response → Ollama response. Field mappings follow the tables in §3.

---

### A.1 — POST /api/generate (non-streaming, text-only)

**Ollama request (input):**
```json
{
  "model": "llama-3.2-3b",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "seed": 42,
    "num_predict": 100
  }
}
```

**OpenAI request (after translation):**
```json
{
  "model": "llama-3.2-3b",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "temperature": 0.7,
  "seed": 42,
  "max_tokens": 100
}
```

**OpenAI response (from upstream):**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1720000000,
  "model": "llama-3.2-3b",
  "choices": [
    {
      "text": "The sky appears blue because of Rayleigh scattering.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 43,
    "total_tokens": 51
  }
}
```

**Ollama response (after translation):**
```json
{
  "model": "llama-3.2-3b",
  "created_at": "2024-07-03T06:26:40Z",
  "response": "The sky appears blue because of Rayleigh scattering.",
  "done": true,
  "done_reason": "stop",
  "total_duration": 0,
  "load_duration": 0,
  "prompt_eval_count": 8,
  "prompt_eval_duration": 0,
  "eval_count": 43,
  "eval_duration": 0
}
```

Translation notes:
- `options.temperature` → `temperature`, `options.seed` → `seed`, `options.num_predict` → `max_tokens`.
- `created` (Unix int) → `created_at` (ISO 8601 string).
- `choices[0].text` → `response`.
- `choices[0].finish_reason` → `done_reason`.
- `usage.prompt_tokens` → `prompt_eval_count`, `usage.completion_tokens` → `eval_count`.
- Duration fields default to `0` (not available from upstream).

---

### A.2 — POST /api/generate (streaming)

**Ollama request:**
```json
{
  "model": "llama-3.2-3b",
  "prompt": "Hello",
  "stream": true,
  "options": {"temperature": 0}
}
```

**OpenAI request (after translation):**
```json
{
  "model": "llama-3.2-3b",
  "prompt": "Hello",
  "stream": true,
  "temperature": 0
}
```

**OpenAI SSE stream (from upstream, three chunks):**
```
data: {"id":"cmpl-1","object":"text_completion.chunk","created":1720000001,"model":"llama-3.2-3b","choices":[{"text":"Hello","index":0,"finish_reason":null}]}

data: {"id":"cmpl-1","object":"text_completion.chunk","created":1720000001,"model":"llama-3.2-3b","choices":[{"text":" there","index":0,"finish_reason":null}]}

data: {"id":"cmpl-1","object":"text_completion.chunk","created":1720000001,"model":"llama-3.2-3b","choices":[{"text":"!","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}

data: [DONE]
```

**Ollama NDJSON stream (output):**
```
{"model":"llama-3.2-3b","created_at":"2024-07-03T06:26:41Z","response":"Hello","done":false}
{"model":"llama-3.2-3b","created_at":"2024-07-03T06:26:41Z","response":" there","done":false}
{"model":"llama-3.2-3b","created_at":"2024-07-03T06:26:41Z","response":"!","done":false}
{"model":"llama-3.2-3b","created_at":"2024-07-03T06:26:41Z","response":"","done":true,"done_reason":"stop","total_duration":0,"load_duration":0,"prompt_eval_count":2,"prompt_eval_duration":0,"eval_count":3,"eval_duration":0}
```

Translation notes:
- Each SSE `data:` line produces one Ollama NDJSON line.
- `choices[0].text` contains the delta text; mapped to `response`.
- `done: false` for all intermediate chunks.
- The final chunk (`finish_reason` present) triggers the accumulation of usage
  stats into the final `done: true` object.
- The final object's `response` is empty (per Ollama convention when streamed).
- `[DONE]` does not produce an output line; it triggers the final object.

---

### A.3 — POST /api/chat (non-streaming, with tool calls)

**Ollama request:**
```json
{
  "model": "llama-3.2-3b",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"}
  ],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

**OpenAI request (after translation):**
```json
{
  "model": "llama-3.2-3b",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"}
  ],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

Note: `messages` and `tools` pass through structurally unchanged.

**OpenAI response:**
```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1720000002,
  "model": "llama-3.2-3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Tokyo\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 15,
    "total_tokens": 60
  }
}
```

**Ollama response (after translation):**
```json
{
  "model": "llama-3.2-3b",
  "created_at": "2024-07-03T06:26:42Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_weather",
          "arguments": {
            "city": "Tokyo"
          }
        }
      }
    ]
  },
  "done": true,
  "done_reason": "tool_calls",
  "total_duration": 0,
  "load_duration": 0,
  "prompt_eval_count": 45,
  "prompt_eval_duration": 0,
  "eval_count": 15,
  "eval_duration": 0
}
```

Translation notes:
- `choices[0].message.role` → `message.role`.
- `choices[0].message.content` (null) → `message.content` (empty string).
- OpenAI's `tool_calls` uses string-encoded `arguments`; Ollama uses
  pre-parsed JSON objects. The translator must `JSON.parse()` the
  `arguments` string. This is the only transformation on tool calls.
- `choices[0].finish_reason` → `done_reason`.

---

### A.4 — POST /api/embed

**Ollama request:**
```json
{
  "model": "all-minilm",
  "input": ["Why is the sky blue?", "Why is grass green?"],
  "truncate": true
}
```

**OpenAI request (after translation):**
```json
{
  "model": "all-minilm",
  "input": ["Why is the sky blue?", "Why is grass green?"]
}
```

**OpenAI response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.01, -0.002, 0.05, 0.047, 0.055, 0.009, 0.105, -0.026]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [-0.01, 0.06, 0.025, -0.006, 0.073, 0.017, 0.09, -0.052]
    }
  ],
  "model": "all-minilm",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

**Ollama response (after translation):**
```json
{
  "model": "all-minilm",
  "embeddings": [
    [0.01, -0.002, 0.05, 0.047, 0.055, 0.009, 0.105, -0.026],
    [-0.01, 0.06, 0.025, -0.006, 0.073, 0.017, 0.09, -0.052]
  ],
  "total_duration": 0,
  "load_duration": 0,
  "prompt_eval_count": 12
}
```

Translation notes:
- `data[].embedding` → `embeddings` (array of arrays).
- `usage.prompt_tokens` → `prompt_eval_count`.

---

### A.5 — GET /api/tags

**OpenAI response from `GET /v1/models`:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.2-3b",
      "object": "model",
      "created": 1720000000,
      "owned_by": "library"
    },
    {
      "id": "mistral-7b",
      "object": "model",
      "created": 1719900000,
      "owned_by": "library"
    }
  ]
}
```

**Ollama response (after translation):**
```json
{
  "models": [
    {
      "name": "llama-3.2-3b",
      "model": "llama-3.2-3b",
      "modified_at": "2024-07-03T06:26:40Z",
      "size": 0,
      "digest": "abc12345",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "",
        "quantization_level": ""
      }
    },
    {
      "name": "mistral-7b",
      "model": "mistral-7b",
      "modified_at": "2024-07-02T02:40:00Z",
      "size": 0,
      "digest": "def67890",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "mistral",
        "families": ["mistral"],
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

Translation notes:
- `data[].id` → `name` and `model`.
- `data[].created` (Unix int) → `modified_at` (ISO 8601).
- `size` is `0` (unavailable from OpenAI models endpoint).
- `digest` is derived from hashing `id` (deterministic, non-cryptographic).
- `details.family` is derived from the first segment of the model name
  before any `-` or `:` character. Falls back to `"unknown"`.
- `details.format` is always `"gguf"` (assumed for llamacpp-server).

---

## Appendix B: SSE-to-NDJSON Streaming State Machine

The stream adapter reads bytes from the upstream SSE connection and yields
Ollama NDJSON lines. It maintains internal state to handle the SSE protocol,
accumulate content, and detect end-of-stream.

### States

```
                    ┌──────────┐
                    │   IDLE   │
                    └────┬─────┘
                         │ line received
                    ┌────▼─────┐
          ┌────────│  READING  │────────┐
          │        └────┬─────┘        │
          │ line is      │ line is      │ line is
          │ "data: ..."  │ empty        │ something else
          │              │              │
    ┌─────▼──────┐  ┌───▼────┐   ┌─────▼──────┐
    │ PROCESS    │  │ FLUSH  │   │   SKIP     │
    │ DATA LINE  │  │ EVENT  │   │ (comment   │
    └─────┬──────┘  └───┬────┘   │  or other) │
          │              │        └─────┬──────┘
          │         ┌────▼─────┐       │
          │         │  IDLE    │◄──────┘
          │         └──────────┘
          │
          │ data is "[DONE]"?
          │
    ┌─────▼──────┐
    │   DONE     │──► Emit final Ollama object, close.
    └────────────┘
          │
          │ data is valid JSON?
          │
    ┌─────▼──────┐
    │ TRANSLATE  │──► Transform chunk, yield NDJSON line.
    │  & EMIT    │──► Accumulate content/usage.
    └─────┬──────┘
          │
          │ (back to READING for next line)
```

### Transitions

| From | Trigger | Action | To |
|---|---|---|---|
| (start) | Connection opened | Initialize accumulators: `content = ""`, `tool_calls = []`, `usage = {}`, `finish_reason = None`. | IDLE |
| IDLE | Read a line from the upstream stream | — | READING |
| READING | Line starts with `data: ` | Strip `data: ` prefix. | PROCESS DATA LINE |
| READING | Line is empty or whitespace-only | (SSE event boundary — no action for Ollama translation, as each data line is already an independent event.) | IDLE |
| READING | Line starts with `:`, `event:`, `id:`, or `retry:` | Skip line (SSE comment or metadata field). | READING |
| READING | Line is anything else not matching above | Skip line (unexpected format). | READING |
| PROCESS DATA LINE | Stripped data equals `[DONE]` | Build final Ollama object with accumulated stats, `done: true`. Yield it. Close connection. | DONE (terminal) |
| PROCESS DATA LINE | Stripped data contains top-level `"error"` key | Build Ollama error NDJSON line with the error message, `done: true`. Yield it. Close connection. | DONE (terminal) |
| PROCESS DATA LINE | Stripped data is valid JSON | TRANSFORM & EMIT (see below). | READING |
| PROCESS DATA LINE | Stripped data is not valid JSON | Log warning. Discard line. | READING |

### TRANSFORM & EMIT subroutine

When a valid JSON SSE chunk is received, apply these steps:

1. **Parse** the JSON object from the `data:` line.
2. **Extract** the delta from the OpenAI chunk structure:
   - For completions: `choices[0].text` (may be null/absent).
   - For chat completions: `choices[0].delta.content` (may be null/absent).
   - For chat tool calls: `choices[0].delta.tool_calls` (array of deltas).
3. **Accumulate:**
   - Append `choices[0].text` or `choices[0].delta.content` to `content`.
   - Merge `choices[0].delta.tool_calls` into `tool_calls` using index-based merging.
   - If chunk has `usage`, replace `usage` accumulator with the new value.
   - If chunk has `choices[0].finish_reason` that is not null, set `finish_reason`.
4. **Build Ollama chunk:**
   - For completions: `{"model": "<model>", "created_at": "<ts>", "response": "<delta text or ''>", "done": false}`.
   - For chat completions: `{"model": "<model>", "created_at": "<ts>", "message": {"role": "assistant", "content": "<delta text or ''>"}, "done": false}`.
   - If tool call deltas are present, include `message.tool_calls` with the accumulated state.
5. **Yield** the serialized JSON line with trailing `\n`. Flush immediately.
6. **If** `finish_reason` is now set (this was the final chunk), build and yield the final `done: true` object:
   ```json
   {"model":"...","created_at":"...","response":"","done":true,"done_reason":"<reason>","total_duration":0,"load_duration":0,"prompt_eval_count":<n>,"prompt_eval_duration":0,"eval_count":<n>,"eval_duration":0}
   ```
   For chat: `message` with accumulated `content` and `tool_calls`.

### Error transitions during reading

| From | Trigger | Action | To |
|---|---|---|---|
| READING | Connection closed (EOF before `[DONE]`) | Emit last accumulated content as an NDJSON line. Emit error NDJSON line with `done: true`. Close. | DONE (terminal) |
| READING | Read timeout (no data within `REQUEST_TIMEOUT`) | Emit last accumulated content. Emit timeout error NDJSON line with `done: true`. Close. | DONE (terminal) |

### Implementation note: line reading

The adapter reads the upstream response as a stream of lines (using an async
iterator that splits on `\n`). Each line is stripped of the trailing `\n`
before processing through the state machine. The SSE specification uses `\r\n`
as the line terminator; the adapter handles both `\n` and `\r\n` by stripping
trailing carriage returns.

---

## References

The following files are available in the repository. They contain the
authoritative API specifications used to derive the contracts in this
document. A model implementing this proxy should read these files to
reproduce exact payloads for unit tests and end-to-end tests.

### Ollama API reference

| File | Content | Use for |
|---|---|---|
| `reference/ollama/docs/api.md` | Complete Ollama REST API documentation. Every endpoint with curl examples, request parameters, and response payloads (both streaming and non-streaming). | Test fixtures: copy the curl request bodies verbatim and assert against the documented response shapes. |
| `reference/ollama/docs/api/streaming.mdx` | Streaming response format specification. | Understand the exact NDJSON line format used by Ollama. |
| `reference/ollama/docs/api/errors.mdx` | Error response format and status codes. | Reproduce exact error payloads in test assertions. |
| `reference/ollama/docs/api/openai-compatibility.mdx` | Documented mapping from Ollama's own OpenAI-compatible layer (reverse direction). Useful for understanding how Ollama itself maps fields when acting as an OpenAI server. | Cross-reference field mappings; confirm which OpenAI fields Ollama actually supports. |
| `reference/ollama/docs/modelfile.mdx` | Valid model parameters for the `options` object (temperature, top_p, top_k, seed, stop, num_predict, num_ctx, etc.). | Ensure all option keys that the translator must handle are covered. |
| `reference/ollama/docs/api/introduction.mdx` | API conventions (model names, durations, streaming). | Validate that response conventions (nanosecond durations, ISO 8601 timestamps) are handled correctly. |

### OpenAI API reference

| File | Content | Use for |
|---|---|---|
| `reference/openai-openapi/openapi.yaml` | Full OpenAI REST API OpenAPI 3.1 specification. Contains request/response schemas for `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`. | Schema validation: verify that translated requests match the expected OpenAI request shape. Verify that upstream responses can be parsed correctly. |
| Sections of interest in `openapi.yaml`: | — | — |
| Line ~3080: `/chat/completions` | Chat completions endpoint (POST + GET for listing). Includes `CreateChatCompletionRequest` and response schemas. | Verify `messages`, `tools`, `stream`, `response_format` fields. |
| Line ~4928: `/completions` | Legacy completions endpoint. Includes `CreateCompletionRequest` and response schemas. | Verify `prompt`, `suffix`, `max_tokens`, `stop` fields. |
| Line ~13221: `/models` | Models listing endpoint. Includes `ListModelsResponse` schema. | Verify `data[].id`, `data[].created`, `data[].owned_by` fields. |
| `/v1/embeddings` section | Embeddings endpoint. Request accepts `model`, `input` (string or array). Response contains `data[].embedding`. | Verify embedding shapes for `/api/embed` translation. |

### Test fixture strategy

When writing tests, use these sources in priority order:

1. **BLUEPRINT.md Appendix A** — use the exact JSON payloads shown as test inputs and expected outputs. These cover the core translation paths.
2. **`reference/ollama/docs/api.md`** — copy curl `-d` bodies as test inputs for edge cases (images, structured outputs, JSON mode, raw mode, empty prompt, suffix, options-heavy requests).
3. **`reference/openai-openapi/openapi.yaml`** — use the response schemas to validate that translated OpenAI requests are well-formed and that upstream responses parse correctly.

For streaming tests:
- Use the SSE chunk sequences shown in Appendix A.2.
- Use `reference/ollama/docs/api.md` streaming examples for chat streaming (including tool call delta accumulation).
- The state machine in Appendix B defines the exact behavior for each SSE line type.

For error tests:
- BLUEPRINT.md §7.1 and §7.2 define all error scenarios and expected response formats.
- `reference/ollama/docs/api/errors.mdx` provides the Ollama-native error format to match.
- Assert that error responses contain `{"error": "..."}` and the correct HTTP status code.

---

