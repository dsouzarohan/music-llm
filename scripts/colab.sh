#!/usr/bin/env bash
# Drives a Colab GPU session for training. See docs/colab.md.
#
#   scripts/colab.sh new                      provision the VM (GPU=A100 by default)
#   colab drivemount -s music                 HUMAN step: mount Drive (interactive)
#   scripts/colab.sh setup                    clone BRANCH (default master), link Drive data, install deps
#   scripts/colab.sh train <config> [flags]   launch train.py detached; log goes to Drive data/logs/
#   scripts/colab.sh logs [n]                 tail the latest training log
#   scripts/colab.sh status                   session status and nvidia-smi
#   scripts/colab.sh fetch <remote> <local>   download a file from the VM
#   scripts/colab.sh stop                     release the VM (billing stops)
#
# Env: SESSION (default music), GPU (default A100), BRANCH (default master).
set -euo pipefail

SESSION="${SESSION:-music}"
GPU="${GPU:-A100}"
BRANCH="${BRANCH:-master}"
REPO_URL="https://github.com/dsouzarohan/music-llm.git"
DRIVE_DATA="/content/drive/MyDrive/Projects/music-llm/data"
VM_REPO="/content/music-llm"

remote() {
  # remote <timeout_s> <bash command>: runs a shell command on the VM, streams output, propagates failure
  colab exec -s "$SESSION" --timeout "$1" --env "CMD=$2" <<'PY'
import os, subprocess, sys
r = subprocess.run(["bash", "-lc", os.environ["CMD"]], capture_output=True, text=True)
print(r.stdout, end="")
print(r.stderr, end="", file=sys.stderr)
if r.returncode:
    raise SystemExit(r.returncode)
PY
}

cmd="${1:-}"
shift || true

case "$cmd" in
  new)
    colab new -s "$SESSION" --gpu "$GPU"
    echo "Next: mount Drive yourself with  ! colab drivemount -s $SESSION"
    ;;
  setup)
    remote 60 "test -d $DRIVE_DATA/tokenized || { echo 'Drive data not found at $DRIVE_DATA: run colab drivemount -s $SESSION first' >&2; exit 1; }"
    remote 120 "if [ -d $VM_REPO/.git ]; then cd $VM_REPO && git fetch -q origin && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH; \
      else git clone -q -b $BRANCH $REPO_URL $VM_REPO; fi && ln -sfn $DRIVE_DATA $VM_REPO/data && cd $VM_REPO && git log --oneline -1"
    # torch is preinstalled on Colab; the rest comes from pyproject.toml
    colab install -s "$SESSION" "miditok==3.0.5.post1" "symusic>=0.5.7,<0.6" "music21>=9.3" "tqdm>=4.67"
    remote 60 "python -c 'import torch, miditok; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'"
    ;;
  train)
    config="${1:?usage: scripts/colab.sh train <config.toml> [train.py flags]}"
    shift
    flags=$(printf ' %q' "$@")
    remote 30 "mkdir -p $VM_REPO/data/logs && cd $VM_REPO && log=data/logs/train_\$(date +%Y%m%d_%H%M%S).log && \
      nohup python -m src.utils.train --config $config $flags > \$log 2>&1 < /dev/null & echo started pid \$! log $VM_REPO/\$log"
    ;;
  logs)
    remote 30 "tail -n ${1:-20} \$(ls -t $VM_REPO/data/logs/train_*.log | head -1)"
    ;;
  status)
    colab status -s "$SESSION"
    remote 30 "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv; pgrep -af 'src.utils.train' || echo 'no training process'"
    ;;
  fetch)
    colab download -s "$SESSION" "${1:?remote path}" "${2:?local path}"
    ;;
  stop)
    colab stop -s "$SESSION"
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
