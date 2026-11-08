# Training on Colab from the terminal

`scripts/colab.sh` wraps the Colab CLI (`google-colab-cli`) for this repo. For the CLI's own
agent guide, run `colab skill`.

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
scripts/colab.sh fetch /content/music-llm/data/runs/<date>/<run>/checkpoints/best.pt data/checkpoints/best.pt
scripts/colab.sh stop                                 # 6. release the VM; billing stops
```

What each step does:

- **setup** checks that Drive is mounted, then clones `BRANCH` into `/content/music-llm` (or
  hard-resets an existing clone to `origin/BRANCH`). It symlinks
  `/content/music-llm/data → MyDrive/Projects/music-llm/data` and installs the non-torch
  dependencies; Colab's preinstalled torch is used. It prints the torch version and GPU name,
  then copies `tokenized/` from Drive to `/content/data-local` (about 60 s for ~20k files).
- **train** runs `python -u -m src.utils.train --config <config> --data_root /content/data-local
  --runs_dir data/runs [flags]` under `nohup`, logging to `data/logs/train_<timestamp>.log`. It
  reads tokens from local disk and writes everything else to Drive: the log and the run directory
  (`data/runs/<YYYY-MM-DD>/<HHMM>-<name>/` with `metrics.jsonl` and `checkpoints/`) survive a VM drop. To continue after a drop: `scripts/colab.sh train <config> --resume data/runs/<date>/<run>` (resumes from `checkpoints/last.pt`).
- Extra flags pass through, e.g. `scripts/colab.sh train configs/colab-186m-jan2026.toml --epochs 2 --profile_steps 10 --no-compile`.

## Gotchas (found in the first run)

- **Drive is slow for many small files.** Reading the 20k token files straight from the Drive
  mount ran at ~0.2 MB/s, so data loading would have taken longer than training. Each file is a
  network round trip; 64 parallel `cp`s hide that latency, so setup stages the corpus locally.
- **Python buffers stdout when it's redirected to a file**, so the log stayed empty. `train`
  uses `python -u`.
- **`a && b & c` backgrounds the whole `a && b` chain**, not just `b`, and that subshell keeps
  the `exec` output pipe open until training ends, leaving the kernel BUSY. The launch command
  wraps only the `nohup` in braces: `a && { nohup b & }`.
- **While the kernel is busy, `colab exec` can't run**, but `colab console` (a separate tmux
  shell) still can. Pipe a command in followed by a short `sleep`, or the console closes before
  printing.
- **`ls | wc -l` undercounts the token files**: some filenames contain newlines. Count with
  Python or `find -print0`.
- **Colab may terminate a session whose notebook looks idle**, even while a detached process
  trains on the GPU. The first baseline attempt died after ~80 min with no browser attached.
  Keep the session open in a browser tab (`colab url -s music --open`) during long runs.
- **Drive uploads asynchronously.** When that VM died, `metrics.jsonl` (small appends) had
  synced through epoch 14, but the ~2 GB `best.pt` writes from epochs 10–14 never finished
  uploading, so Drive's `best.pt` was from epoch 9. A file existing on Drive doesn't mean its
  latest write did. Resume from the newest checkpoint whose `epoch` matches what you expect.
- **`colab exec` connections drop occasionally** (`RuntimeError: Connection was lost`). Every
  `setup` step is idempotent, so re-run it. `colab console` was steadier for one-off commands.
- **`colab new` can stop for a browser auth code** when its saved login lapses or it asks for
  new scopes (it requested `drive.file`). Run it from your own terminal.

## Constraints that shaped the script

- **`colab exec -f file.py` sends one file's contents** to the remote kernel, which has no copy
  of the repo. So setup clones the repo onto the VM.
- **`colab drivemount` needs a human at a TTY**, so an agent can't run it. Every other step is
  scripted.
- **`exec` has a default timeout of 30 s.** Training is launched detached and polled through
  the log, never run inside a blocking `exec`.
- **Billing.** Sessions bill until `colab stop`. A100 runs of the 186M model take hours.
- `colab run --gpu T4 script.py` provisions, runs one script and tears down. It's handy for
  single-file experiments but can't see `src/`.
