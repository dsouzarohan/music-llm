#!/usr/bin/env bash
# Creates .venv from pyproject.toml + uv.lock, including the dev group.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked "$@"
