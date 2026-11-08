#!/usr/bin/env bash
# Drives a Colab GPU session for training. See docs/colab.md.
#
#   scripts/colab.sh new                      provision the VM (GPU=A100 by default)
#   colab drivemount -s music                 HUMAN step: mount Drive (interactive)
#   scripts/colab.sh setup                    clone BRANCH (default master), link Drive data, install deps,
#                                             copy tokenized/ from Drive to local disk
#   scripts/colab.sh train <config> [flags]   launch train.py detached; log goes to Drive data/logs/
#   scripts/colab.sh queue <queue file>       run a queue file (scripts/queue.sh) detached; same log folder
#   scripts/colab.sh logs [n]                 tail the latest training log
#   scripts/colab.sh status                   session status and nvidia-smi
#   scripts/colab.sh fetch <remote> <local>   download a file or folder from the VM (scp -r)
#   scripts/colab.sh ssh [command]            shell on the VM, or run one command there
#   scripts/colab.sh stop                     release the VM (billing stops)
#
# Env: SESSION (default music), GPU (default A100), BRANCH (default master).
# Commands reach the VM over SSH through `colab ssh --proxy-mode` (key: ~/.ssh/id_ed25519).
set -euo pipefail

SESSION="${SESSION:-music}"
GPU="${GPU:-A100}"
BRANCH="${BRANCH:-master}"
REPO_URL="https://github.com/dsouzarohan/music-llm.git"
DRIVE_DATA="/content/drive/MyDrive/Projects/music-llm/data"
VM_REPO="/content/music-llm"
LOCAL_DATA="/content/data-local"

SSH_OPTS=(-o "ProxyCommand=colab ssh --proxy-mode -s $SESSION" -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30)
SSH_HOST="root@colab-$SESSION"

remote() {
  # remote <timeout_s> <bash command>: runs a shell command on the VM, streams output, propagates failure
  ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$SSH_HOST" "timeout $1 bash -lc $(printf %q "$2")"
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
    remote 600 "pip install -q 'miditok==3.0.5.post1' 'symusic>=0.5.7,<0.6' 'music21>=9.3' 'tqdm>=4.67'"
    remote 60 "python -c 'import torch, miditok; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'"
    # Drive reads are slow per file; 64 parallel copies hide the latency. --update=none skips files already staged.
    remote 1800 "cd $DRIVE_DATA/tokenized && mkdir -p $LOCAL_DATA/tokenized/train-aug $LOCAL_DATA/tokenized/val && \
      cp -r --update=none config $LOCAL_DATA/tokenized/ && start=\$(date +%s) && \
      find train-aug val -name '*.json' -print0 | xargs -0 -P 64 -I{} cp --update=none {} $LOCAL_DATA/tokenized/{} && \
      echo staged \$(find $LOCAL_DATA/tokenized/train-aug -name '*.json' -printf . | wc -c) train / \$(find $LOCAL_DATA/tokenized/val -name '*.json' -printf . | wc -c) val files in \$((\$(date +%s) - start))s"
    ;;
  train)
    config="${1:?usage: scripts/colab.sh train <config.toml> [train.py flags]}"
    shift
    flags=$(printf ' %q' "$@")
    # braces keep `&` scoped to nohup; -u unbuffers stdout so the log fills live
    remote 30 "mkdir -p $VM_REPO/data/logs && cd $VM_REPO && log=data/logs/train_\$(date +%Y%m%d_%H%M%S).log && \
      { nohup python -u -m src.utils.train --config $config --data_root $LOCAL_DATA --runs_dir data/runs $flags > \$log 2>&1 < /dev/null & } && \
      echo started pid \$! log $VM_REPO/\$log"
    ;;
  queue)
    queue="${1:?usage: scripts/colab.sh queue <queue file>}"
    remote 30 "mkdir -p $VM_REPO/data/logs && cd $VM_REPO && log=data/logs/train_\$(date +%Y%m%d_%H%M%S).log && \
      { PYTHON=python DATA_ROOT=$LOCAL_DATA RUNS_DIR=data/runs nohup scripts/queue.sh $(printf %q "$queue") > \$log 2>&1 < /dev/null & } && \
      echo started pid \$! log $VM_REPO/\$log"
    ;;
  logs)
    remote 30 "tail -n ${1:-20} \$(ls -t $VM_REPO/data/logs/train_*.log | head -1)"
    ;;
  status)
    colab status -s "$SESSION"
    remote 30 "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv; pgrep -af 'src.utils|queue.sh' || echo 'no training process'"
    ;;
  fetch)
    scp -r "${SSH_OPTS[@]}" "$SSH_HOST:${1:?remote path}" "${2:?local path}"
    ;;
  ssh)
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"
    ;;
  stop)
    colab stop -s "$SESSION"
    ;;
  *)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
