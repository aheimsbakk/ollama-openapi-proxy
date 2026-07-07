A middleware program.

Converting api request from an Ollama compatible API to an OpenAI compatible API.

Program exposes the Ollama API. Converts all incomming API calls and redirects them to an known OpenAI API endpoint.

Goal. llamacpp-server exposes the OpenAPI endpoint. This progam exposes Ollama API and redirects to llamacpp-server. We can now use llamacpp-server as it was ollama, and connect it to Ollama only tools.

OpenAI API reference: reference/openai-openapi/openapi.yaml
Ollama API reference: reference/ollama/docs/api.md, reference/ollama/docs/api/* 

Python:
  * flask or fastapi
  * venv with uv
  * installable with uv
  * prefer Python builtin above external libraries
  * use builtin for argument parsing