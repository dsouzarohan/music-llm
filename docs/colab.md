# Training on Colab from the terminal

Uses the Colab CLI (`google-colab-cli`, v0.7.4 installed). For the CLI's own agent guide, run
`colab skill`. This doc only covers what's specific to this repo.

> **Status:** written from the CLI's help and README. The end-to-end flow below has **not been
> run yet**. Update this doc the first time it is.

## One-time setup

```bash
uv tool install google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab sessions          # verifies auth
```

## Constraints that shape the workflow

- **`colab exec -f file.py` sends one file's contents** to the remote kernel. The kernel has no
  copy of the repo, so `from src.model...` fails. The repo has to exist on the VM.
- **The real dataset lives on Drive** at `MyDrive/Projects/music-llm/data`, and so do earlier
  copies of the code: the journal shows `/content/drive/MyDrive/Projects/music-llm/src/...`.
  `colab drivemount` needs a human at a TTY, so an agent can't run it.
- **`exec` has a default timeout of 30 s.** Don't run a multi-hour training through a blocking
  `exec`. Start it detached and poll the log instead.
- **Billing.** Sessions bill until `colab stop`. A100 runs of the 186M model took hours.

## Workflow

```bash
# 1. Provision (agent can do this)
colab new -s music --gpu A100

# 2. Mount Drive (HUMAN: interactive)
! colab drivemount -s music

# 3. Get current code onto the VM (agent): clone fresh, symlink Drive data
echo 'cd /content && git clone https://github.com/dsouzarohan/music-llm.git \
  && ln -s /content/drive/MyDrive/Projects/music-llm/data /content/music-llm/data' | colab console -s music
colab install -s music miditok==3.0.5.post1 "symusic>=0.5.7,<0.6" music21 tqdm   # torch is preinstalled

# 4. Start training detached, logging to Drive so it survives a VM drop
echo 'cd /content/music-llm && nohup python -m src.utils.train --data_root data \
  --checkpoint_dir data/checkpoints --vocab_size 3500 --block_size 2048 --n_layer 24 \
  --n_head 12 --embedding_dim 768 --dropout 0.1 --batch_size 128 --micro_batch_size 8 \
  --learning_rate 1e-4 --epochs 50 --save_every 5 > data/logs/train_$(date +%Y%m%d_%H%M).log 2>&1 &' \
  | colab console -s music

# 5. Poll
echo 'import subprocess; print(subprocess.run("tail -n 5 /content/music-llm/data/logs/train_*.log", shell=True, capture_output=True, text=True).stdout)' | colab exec -s music
colab status -s music

# 6. Pull artifacts and release the VM
colab download -s music /content/music-llm/data/checkpoints/best.pt ./data/checkpoints/best.pt
colab stop -s music
```

Notes:

- If the repo is private, step 3 needs a token or deploy key. The alternative is to
  `git pull` inside the Drive copy.
- `colab run --gpu T4 script.py` provisions a VM, runs one script and tears it down. It's
  handy for single-file experiments, but it can't see `src/` either.
- The Colab MCP server (`googlecolab/colab-mcp`) is a separate tool for agent help *inside*
  notebooks. It isn't needed for this terminal flow.
