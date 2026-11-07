# Training on Colab from the terminal

`scripts/colab.sh` wraps the Colab CLI (`google-colab-cli`) for this repo. For the CLI's own
agent guide, run `colab skill`.

> **Status:** scripted in T3–T5. The first end-to-end run is the T5 baseline; correct this doc
> with anything that run turns up.

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
scripts/colab.sh fetch /content/music-llm/data/runs/<run>/best.pt data/checkpoints/best.pt
scripts/colab.sh stop                                 # 6. release the VM; billing stops
```

What each step does:

- **setup** checks that Drive is mounted, then clones `BRANCH` into `/content/music-llm` (or
  hard-resets an existing clone to `origin/BRANCH`). It symlinks
  `/content/music-llm/data → MyDrive/Projects/music-llm/data` and installs the non-torch
  dependencies; Colab's preinstalled torch is used. It finishes by printing the torch version
  and GPU name.
- **train** runs `python -m src.utils.train --config <config> [flags]` under `nohup`, logging to
  `data/logs/train_<timestamp>.log`. Because `data/` is the Drive symlink, the log, the run
  directory (`data/runs/<timestamp>-<name>/` with `metrics.jsonl` and checkpoints) all survive
  a VM drop. To continue after a drop: `scripts/colab.sh train <config> --resume data/runs/<run>/epoch_N.pt`.
- Extra flags pass through, e.g. `scripts/colab.sh train configs/colab-186m-jan2026.toml --epochs 2 --profile_steps 10 --no-compile`.

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
