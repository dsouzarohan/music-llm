# Training on Colab from the terminal

`scripts/colab.sh` wraps the Colab CLI (`google-colab-cli`, ≥ 0.7.5) for this repo. For the
CLI's own agent guide, run `colab skill`.

Since 2026-11-08 every remote command goes over **SSH**: `colab ssh --proxy-mode -s <session>` is
used as ssh's `ProxyCommand`, tunnelling SSH through Colab's websocket, so plain `ssh`/`scp` work
(key: `~/.ssh/id_ed25519`, user `root`). Earlier versions sent commands through the notebook
kernel with `colab exec`, which hung (see Gotchas). `scripts/colab.sh ssh` opens a shell on the VM.

> **Status:** run end to end on 2026-11-07 (T5) on an A100-SXM4-40GB, torch 2.14.1+cu130.
> The lessons from that run are under "Gotchas" below.

## One-time setup

```bash
uv tool install google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab sessions          # verifies auth
```

## Workflow

```bash
scripts/colab.sh new                                  # 1. provision an A100 (GPU=L4 etc. to change)
! colab drivemount -s music                           # 2. HUMAN: mount Drive (interactive)
BRANCH=master scripts/colab.sh setup                  # 3. clone, link Drive data, install deps
scripts/colab.sh train configs/colab-186m-jan2026.toml  # 4. launch detached
scripts/colab.sh logs 20                              # 5. poll
scripts/colab.sh status                               #    GPU utilisation, is training alive
scripts/colab.sh fetch /content/runs/<date>/<run> data/runs/<date>/   # scp -r
scripts/colab.sh stop                                 # 6. release the VM; billing stops
```

What each step does:

- **setup** checks that Drive is mounted, then clones `BRANCH` into `/content/music-llm` (or
  hard-resets an existing clone to `origin/BRANCH`). It symlinks
  `/content/music-llm/data → MyDrive/Projects/music-llm/data` and `pip install`s the non-torch
  dependencies; Colab's preinstalled torch is used. It prints the torch version and GPU name,
  then copies `tokenized/` from Drive to `/content/data-local` (62 s on 2026-11-07, 321 s on
  2026-11-08: Drive's speed varies).
- **train** runs `python -u -m src.utils.train --config <config> --data_root /content/data-local
  --runs_dir /content/runs [flags]` under `nohup`. Runs and the log (`/content/logs/train_<timestamp>.log`)
  live on the VM's local disk; `scripts/sync-runs.sh` copies `/content/runs` and `/content/logs` to
  Drive's `data/runs` and `data/logs` every 5 minutes while the job runs, and once more at the end
  (`*.tmp` files skipped). If the Drive mount drops, training continues and the copy retries; if
  the VM dies, Drive has everything up to the last copy.
- **To continue after a dropped VM:** on a new session, `scripts/colab.sh pull <YYYY-MM-DD>/<run>`
  copies the run from Drive to local disk, then train or queue it with `--resume
  /content/runs/<YYYY-MM-DD>/<run>` (resumes from `checkpoints/last.pt`).
- **queue** runs queue files with `scripts/queue.sh`, launched and synced the same way. Each line
  of a queue file is a `python -m` command (train or sample) that may use `$DATA_ROOT` (the staged
  corpus) and `$RUNS_DIR` (`/content/runs`).
  Lines run one after another and a failed line doesn't stop the rest. Queue files live in
  `configs/queues/`, e.g. `scripts/colab.sh queue configs/queues/t5-ablations.txt`.
- Extra flags pass through, e.g. `scripts/colab.sh train configs/colab-186m-jan2026.toml --epochs 2 --profile_steps 10 --no-compile`.

## Gotchas

- **`colab exec` hung twice in a row** (2026-11-08, CLI 0.7.4): once waiting 28 minutes on a
  120 s timeout without the command reaching the VM, once raising `Connection was lost` and then
  never exiting. `colab console` kept working, so the kernel channel was the problem. Fixed by
  moving `colab.sh` to SSH.
- **The Drive mount can vanish while the VM keeps running** (2026-11-08, 13:34): `os.replace` of
  `best.pt.tmp` failed with `FileNotFoundError` and every later queue line failed at `mkdir`, because
  `data` pointed at a mount that was gone. Hence training on local disk with a periodic sync.
- **`colab ssh` creates a runtime when the session is missing.** After the A100 was released, a
  status check over SSH silently started a CPU runtime under the same name. `colab.sh` now checks
  `colab sessions` before every SSH/scp call and stops if the session is gone.
- **A session can disappear mid-run even with the browser tab open** (2026-11-08: the A100 went
  between two checks 30 min apart, ~1 h after creation). Save often (`save_every`) and resume.
- **`colab new` failing with a 503 traceback** means the allocator timed out (~60 s) handing out
  that GPU type: the A100 pool was full while an L4 came up in 5 s. Try another `GPU=`.

- **Drive is slow for many small files.** Reading the 20k token files straight from the Drive
  mount ran at ~0.2 MB/s, so data loading would have taken longer than training. Each file is a
  network round trip; 64 parallel `cp`s hide that latency, so setup stages the corpus locally.
- **Python buffers stdout when it's redirected to a file**, so the log stayed empty. `train`
  uses `python -u`.
- **`a && b & c` backgrounds the whole `a && b` chain**, not just `b`, and that subshell keeps
  the `exec` output pipe open until training ends, leaving the kernel BUSY. The launch command
  wraps only the `nohup` in braces: `a && { nohup b & }`.
- **`ls | wc -l` undercounts the token files**: some filenames contain newlines. Count with
  Python or `find -print0`.
- **Colab may terminate a session whose notebook looks idle**, even while a detached process
  trains on the GPU. The first baseline attempt died after ~80 min with no browser attached.
  Keep the session open in a browser tab (`colab url -s music --open`) during long runs.
- **Drive uploads asynchronously** (before the local-disk layout). When that VM died, `metrics.jsonl` (small appends) had
  synced through epoch 14, but the ~2 GB `best.pt` writes from epochs 10–14 never finished
  uploading, so Drive's `best.pt` was from epoch 9. A file existing on Drive doesn't mean its
  latest write did. Resume from the newest checkpoint whose `epoch` matches what you expect.
- **Every `setup` step is idempotent**, so re-run it after any dropped connection.
- **`colab new` can stop for a browser auth code** when its saved login lapses or it asks for
  new scopes (it requested `drive.file`). Run it from your own terminal.

## Constraints that shaped the script

- **The VM has no copy of the repo**, so setup clones `BRANCH` from GitHub. Push before setup.
- **`colab drivemount` needs a human at a TTY**, so an agent can't run it. Every other step is
  scripted.
- **Training is launched detached** (`nohup`, output to a log on Drive) and polled through the
  log, so a dropped SSH connection doesn't stop it.
- **Billing.** Sessions bill until `colab stop`. A100 runs of the 186M model take hours.
- `colab run --gpu T4 script.py` provisions, runs one script and tears down. It's handy for
  single-file experiments but can't see `src/`.
