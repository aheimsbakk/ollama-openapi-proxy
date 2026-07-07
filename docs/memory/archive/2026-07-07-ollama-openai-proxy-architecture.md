---
topic: "Ollama-to-OpenAI proxy architecture"
importance: high
category: decision
tags: [architecture, blueprint, proxy, ollama, openai, api-translation]
created: 2026-07-07T15:47:06Z
model: opencode/deepseek-v4-pro
---

The middleware exposes Ollama API endpoints and translates them to OpenAI-compatible upstream calls. It is fully stateless with no persistence. Primary inference endpoints (/api/generate, /api/chat, /api/embed) are fully mapped. Model management endpoints (/api/create, /api/pull, etc.) return 501. Configuration is via environment variables: UPSTREAM_URL, LISTEN_HOST, LISTEN_PORT, REQUEST_TIMEOUT. Streaming uses SSE-to-NDJSON conversion.
