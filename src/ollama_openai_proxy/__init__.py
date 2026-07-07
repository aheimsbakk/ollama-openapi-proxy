"""Ollama-to-OpenAI API translation proxy."""

from __future__ import annotations

import importlib.metadata
import pathlib

try:
    __version__ = importlib.metadata.version("ollama-openai-proxy")
except importlib.metadata.PackageNotFoundError:
    _version_file = pathlib.Path(__file__).resolve().parent.parent.parent / "VERSION"
    __version__ = _version_file.read_text().strip()
