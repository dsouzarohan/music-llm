#!/usr/bin/env bash
# Runs queue files in order: one `python -u -m <line>` per non-blank, non-# line.
# Lines may use $DATA_ROOT and $RUNS_DIR. A failed line is reported and the queue continues.
#   scripts/queue.sh configs/queues/t3-t4-measure.txt configs/queues/t5-ablations.txt
# Env: PYTHON (default "uv run python"), DATA_ROOT (default data), RUNS_DIR (default $DATA_ROOT/runs).
set -uo pipefail
cd "$(dirname "$0")/.."

[[ $# -ge 1 ]] || { echo "usage: scripts/queue.sh <queue file>..." >&2; exit 2; }
PYTHON="${PYTHON:-uv run python}"
export DATA_ROOT="${DATA_ROOT:-data}"
export RUNS_DIR="${RUNS_DIR:-$DATA_ROOT/runs}"

failed=0
for queue in "$@"; do
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  eval "args=($line)"
  echo "=== $(date '+%F %T') ${args[*]}"
  if ! $PYTHON -u -m "${args[@]}" < /dev/null; then
    echo "=== FAILED: ${args[*]}"
    failed=$((failed + 1))
  fi
done < "$queue"
done
echo "=== queue done $(date '+%F %T'), $failed failed"
exit $((failed > 0))
