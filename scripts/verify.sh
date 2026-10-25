#!/usr/bin/env bash
# The gate: unit tests plus an end-to-end smoke train. Extra arguments go to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run pytest "$@"
