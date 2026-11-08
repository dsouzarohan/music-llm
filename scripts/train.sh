#!/usr/bin/env bash
# Trains from a TOML run spec; extra arguments are train.py flags and override the spec.
#   scripts/train.sh configs/smoke.toml --data_root /tmp/synthetic
set -euo pipefail
cd "$(dirname "$0")/.."

config="${1:?usage: scripts/train.sh <config.toml> [train.py flags]}"
shift
uv run python -m src.utils.train --config "$config" "$@"
