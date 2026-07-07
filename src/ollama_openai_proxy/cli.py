"""CLI argument parsing and application entry point."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from ollama_openai_proxy.config import Config, load_config
from ollama_openai_proxy.server import create_app


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments and return a Config instance.

    CLI flags take precedence over environment variables.
    """
    parser = argparse.ArgumentParser(
        prog="ollama-openai-proxy",
        description="Ollama-to-OpenAI API translation proxy",
    )

    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default=None,
        help="Network interface to bind (default: 0.0.0.0, env: LISTEN_HOST)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="TCP port to bind (default: 11434, env: LISTEN_PORT)",
    )
    parser.add_argument(
        "--upstream-url",
        "-u",
        type=str,
        default=None,
        help="Base URL of the OpenAI-compatible upstream server (default: http://localhost:8080/v1, env: UPSTREAM_URL)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=None,
        help="Maximum time in seconds to wait for an upstream response (default: 300, env: REQUEST_TIMEOUT)",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity: -v=WARNING, -vv=INFO, -vvv=DEBUG (default: ERROR)",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Print the program version and exit",
    )

    args = parser.parse_args(argv)

    if args.version:
        from ollama_openai_proxy import __version__

        print(f"ollama-openai-proxy {__version__}")
        sys.exit(0)

    return load_config(
        host=args.host,
        port=args.port,
        upstream_url=args.upstream_url,
        timeout=args.timeout,
        verbosity=args.verbosity,
    )


def setup_logging(verbosity: int) -> None:
    """Configure the root logger based on verbosity level.

    Verbosity mapping:
    - 0 (default): ERROR
    - 1 (-v): WARNING
    - 2 (-vv): INFO
    - 3 (-vvv): DEBUG
    """
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    level = level_map.get(verbosity, logging.ERROR)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> None:
    """Application entry point.

    Parses arguments, configures logging, creates the FastAPI app,
    and starts uvicorn.
    """
    config = parse_args(argv)
    setup_logging(config.verbosity)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.listen_host,
        port=config.listen_port,
        log_level="critical",  # We use our own logging setup
    )
