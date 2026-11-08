#!/usr/bin/env bash
# Runs a queue file: one `python -u -m <line>` per non-blank, non-# line, in order.
# Lines may use $DATA_ROOT and $RUNS_DIR. A failed line is reported and the queue continues.
#   scripts/queue.sh configs/queues/t5-ablations.txt
# Env: PYTHON (default "uv run python"), DATA_ROOT (default data), RUNS_DIR (default $DATA_ROOT/runs).
set -uo pipefail
cd "$(dirname "$0")/.."

queue="${1:?usage: scripts/queue.sh <queue file>}"
PYTHON="${PYTHON:-uv run python}"
export DATA_ROOT="${DATA_ROOT:-data}"
export RUNS_DIR="${RUNS_DIR:-$DATA_ROOT/runs}"

failed=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  eval "args=($line)"
  echo "=== $(date '+%F %T') ${args[*]}"
  if ! $PYTHON -u -m "${args[@]}" < /dev/null; then
    echo "=== FAILED: ${args[*]}"
    failed=$((failed + 1))
  fi
done < "$queue"
echo "=== queue done $(date '+%F %T'), $failed failed"
exit $((failed > 0))
