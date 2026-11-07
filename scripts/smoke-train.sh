#!/usr/bin/env bash
# Runs configs/smoke.toml on a synthetic corpus in a temp dir.
# Extra arguments go to train.py and override the config, e.g.
#   scripts/smoke-train.sh --run_dir /tmp/smoke
#   scripts/smoke-train.sh --data_root data --vocab_size 3000   (real tokens instead of synthetic)
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [[ " $* " != *" --data_root "* ]]; then
  uv run python -m tests.synthetic "$tmp/data"
fi

scripts/train.sh configs/smoke.toml --data_root "$tmp/data" "$@"
