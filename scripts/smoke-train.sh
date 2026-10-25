#!/usr/bin/env bash
# Tiny 1-epoch run of train.py on a synthetic corpus, in a temp dir.
# Extra arguments go to train.py and override the defaults below, e.g.
#   scripts/smoke-train.sh --checkpoint_dir /tmp/ckpt
#   scripts/smoke-train.sh --data_root data --vocab_size 3000   (real tokens instead of synthetic)
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

defaults=(
  --data_root "$tmp/data" --checkpoint_dir "$tmp/ckpt"
  --vocab_size 64 --block_size 64 --n_layer 2 --n_head 2 --embedding_dim 64
  --batch_size 8 --micro_batch_size 8 --epochs 1
)

if [[ " $* " != *" --data_root "* ]]; then
  uv run python -m tests.synthetic "$tmp/data"
fi

uv run python -m src.utils.train "${defaults[@]}" "$@"
