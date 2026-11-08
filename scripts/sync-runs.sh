#!/usr/bin/env bash
# Copies folders to their destinations every INTERVAL seconds while PID runs, then once more.
# A failed copy (e.g. the Drive mount dropped) is reported and retried next round.
#   scripts/sync-runs.sh <pid> <interval_s> <src>:<dst> [<src>:<dst> ...]
set -u
pid="${1:?pid}"; interval="${2:?interval}"; shift 2

sync_all() {
  for pair in "$@"; do
    src="${pair%%:*}"; dst="${pair#*:}"
    mkdir -p "$dst" 2>/dev/null && rsync -a --exclude '*.tmp' "$src/" "$dst/" \
      || echo "$(date '+%F %T') sync failed: $src -> $dst (retrying in ${interval}s)"
  done
}

while kill -0 "$pid" 2>/dev/null; do
  sync_all "$@"
  sleep "$interval"
done
sync_all "$@"
echo "$(date '+%F %T') final sync done"
