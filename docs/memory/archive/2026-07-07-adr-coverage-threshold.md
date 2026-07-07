---
topic: "ADR-001: Test coverage threshold and scope"
importance: medium
category: decision
tags: [adr, coverage, testing, quality-gate]
created: 2026-07-07T18:08:37Z
---

ADR-001 recorded in docs/adr/adr-001-test-coverage.md. 80% minimum coverage on core modules only. Excludes __init__.py, __main__.py, cli.py, and config.py. Tests must exercise real business logic.
