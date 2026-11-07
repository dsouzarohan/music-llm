# AGENTS.md

Guide for coding agents working in this repo. Read this first. Then read `docs/PLAN.md` for
the current work, and `docs/README.md` for the full docs index.

## What this is

A decoder-only **music transformer**: a language model whose language is music. It generates
guitar MIDI. The pipeline has four stages:

1. Extract the guitar tracks from the Lakh MIDI Clean dataset.
2. Tokenize them with miditok REMI + BPE.
3. Train a GPT-style model with Huang-style relative attention.
4. Sample tokens and decode them back to MIDI.

This is a learning project. When making a change, explain the mechanics behind it and keep
diffs readable.

## How work is organised

- **`docs/ROADMAP.md`** holds the long-term phases, one workstream each.
- **`docs/PLAN.md`** is the phase in progress, written as one story with numbered tasks.
  - Work tasks in order unless the plan says otherwise.
  - Tick the acceptance criteria as they're met.
  - Set the task's status when it's done.
  - When the story is done, the next roadmap phase becomes the new `PLAN.md`.
  - **One PLAN ships as one PR.** Commit task by task on one branch, keep the PR open until
    the whole story is done (including any cloud run whose results the plan records), then
    squash-merge. The PR title is the squash commit, so write it as a Conventional Commit.
- **`docs/known-issues.md`** lists bugs with file:line. Tasks reference issues by number.
  When you fix one, mark it fixed there.
- **`docs/journal/`** is the dated project log, one file per entry. Add an entry when a session
  produces results worth remembering, such as a training run, a finding or a decision.

## Harness principles

- **Agent-agnostic.** Everything another agent would need to reuse lives in the repo, not in
  agent-specific hidden folders (`.claude/`, `.codex/`, `.cursor/`). That includes docs,
  scripts and, later, skills, which will go in a top-level `skills/` folder.
- **Script-driven.** Routine actions are scripts under `scripts/`: setup, verification, smoke
  training, data reports. Agents run the scripts instead of retyping commands.
  `scripts/verify.sh` is the gate: run it before calling any code change done. CI runs the
  same script on every PR.
- **Docs follow code.** If a change makes a doc wrong (architecture, data pipeline, commands
  in this file), update the doc in the same change.

## Environment (uv, not conda)

```bash
uv sync                      # creates .venv from pyproject.toml + uv.lock (Python 3.12)
uv run python -m src.utils.train --help
uv run jupyter lab           # playground notebooks (dev group, installed by default)
```

- Use `uv add <pkg>` or `uv add --dev <pkg>` for dependencies, never `pip install`.
- `miditok` is pinned exactly (`==3.0.5.post1`), because the saved `tokenizer.json` format
  and the `augment_dataset` API are tied to it.
- The old conda env `music-llm` is legacy. Don't use it or update it.

## Layout

```
src/model/model.py                        MusicTransformer, RelativeHead (+ unused Head)
src/utils/train.py                        training entry point: parse_args() + train(args)
src/utils/checkpoint.py                   save_checkpoint / load_checkpoint (one schema)
src/utils/config.py                       TOML run spec -> argparse defaults
src/utils/metrics.py                      FLOPs/token, GPU peak FLOPs, MFU
src/utils/hyperparameters.py              default hparams (CLI defaults for train.py)
configs/*.toml                            run specs (smoke, colab-186m-jan2026)
src/utils/midi_extractor.py               Lakh MIDI -> per-guitar-track .mid (music21)
src/utils/tokenizer.py                    split -> augment -> train BPE -> tokenize to JSON
src/utils/data/random_guitar_seq_dataset.py   default dataset: random windows
src/utils/data/guitar_dataset.py          sliding-window dataset (--use_guitar_ds)
playground/*.ipynb                        exploration + inference (model_run.ipynb)
tests/                                    pytest suite; tests/synthetic.py writes a fake token corpus
scripts/setup.sh                          uv sync --locked
scripts/verify.sh                         the gate: pytest, including an end-to-end smoke train
scripts/train.sh                          train from a config: scripts/train.sh <config.toml> [flags]
scripts/smoke-train.sh                    configs/smoke.toml on a synthetic corpus
scripts/colab.sh                          Colab session: new/setup/train/logs/status/fetch/stop
.github/workflows/verify.yml              CI: setup.sh + verify.sh on ubuntu
scripts/stats.sh                          grep/awk snippets for the extraction log (legacy, to be replaced; leave as is)
docs/                                     plan, roadmap, architecture, journal; see docs/README.md
data/                                     gitignored, see "Data" below
```

## Commands

Run everything **from the repo root**. Modules import `src.*`, so `python -m` from the root
is the only invocation that resolves cleanly.

```bash
# Train from a run spec; flags override it. Output goes to data/runs/<timestamp>-<name>/
scripts/train.sh configs/smoke.toml --data_root data --vocab_size 3000
scripts/train.sh configs/smoke.toml --resume data/runs/<run>/epoch_5.pt   # exact resume

# Real runs happen on Colab: see docs/colab.md
scripts/colab.sh train configs/colab-186m-jan2026.toml

# Tiny smoke run on the local snapshot instead of synthetic tokens
scripts/smoke-train.sh --data_root data --vocab_size 3000
```

`tokenizer.py` and `midi_extractor.py` are **destructive**:

- `tokenizer.py` deletes `data/splits`, `train-midi`, `val-midi`, `augmented` and `tokenized`
  at import time.
- The tokenizer's split is unseeded, so a rerun produces a different split.

Don't run either one unless the task is explicitly to regenerate data.

## Verification

```bash
scripts/setup.sh                 # once, or after pulling dependency changes
scripts/verify.sh                # full gate, ~10 s locally
scripts/verify.sh -k causal      # extra args go to pytest
scripts/smoke-train.sh           # just the tiny training run, output to the terminal
```

- Tests run on CPU with a tiny model and synthetic tokens from `tests/synthetic.py`. They
  never read `data/`, so they behave the same on a fresh clone and in CI.
- Known bugs have `xfail(strict=True)` tests that name their issue number. When a fix makes
  one pass, pytest reports it as a failure (XPASS): remove the marker in the same change, and
  mark the issue fixed in `docs/known-issues.md`.
- A new behaviour or bug fix comes with a test. The parameter-count test pins the formula in
  `docs/architecture/model-and-training.md`, so a change to the model's parameters updates
  both.

## Where things run

**Extraction, tokenization and training all run on cloud GPUs (Colab)**, not locally. See
`docs/colab.md`. Local machines are for editing code, smoke tests and tiny-model debugging:

- an Apple Silicon Mac (MPS)
- a Linux box with an RTX 5070 Ti 16 GB (journal paths `/home/piffs/...`)

## Data

`data/` is gitignored. The canonical corpus lives on Google Drive at
`MyDrive/Projects/music-llm/data`, and that is the only copy the pipeline updates. It holds:

- 143 artists
- 8,410 guitar MIDIs
- about 49M training tokens
- vocab 3500

The local `data/` is a **frozen, stale snapshot** from before extraction moved to the VM:

- 2 artists (Green Day, Guns N' Roses)
- about 1.7M training tokens
- vocab 3000

It will not be updated. It's fine for smoke tests. Don't treat its sizes, splits or tokenizer
as representative, and don't "fix" it to match Drive.

## Conventions

- Comments say **what**, not **why**. Rationale belongs in the commit or PR description.
- Docstrings describe behaviour.
- Conventional Commits. No `Co-Authored-By: Claude` trailer and no "Generated with Claude Code"
  line.
- When a change affects training dynamics, put before/after `val_loss` and `ppl` in the
  commit message, with the config used. Compare val losses only on the fixed val set
  (post-T4 runs); earlier numbers used random val windows.
- **Changes that invalidate checkpoints:** some architecture changes make old checkpoints
  unusable, for example init, activation, norm placement, or new parameters. Call that out
  explicitly. The owner resumes runs from `best.pt`, so they need to know.
- Keep `src/utils/hyperparameters.py` as the single source of CLI defaults.
