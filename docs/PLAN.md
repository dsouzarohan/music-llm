# Plan: Phase 1, Foundations

**Story:** make the training code trustworthy and verifiable before changing what it does.
After this phase:

- every change is checked by one script, locally and in CI;
- sampling and resuming work;
- every run leaves a config and a metrics log behind;
- there is a measured baseline that Phase 2 can be compared against.

**Status legend:** `todo` · `in progress` · `done`. Issue numbers (#n) refer to
`docs/known-issues.md`.

**Out of scope for this phase:**

- Architecture changes. RoPE, the new attention path and weight-decay groups (#22) are
  Phase 2.
- Data pipeline changes (#8, #9, #11–#17) and `scripts/stats.sh` are Phase 3.
- Per-head performance work (#23).

Bug fixes here may shift training dynamics slightly. That's expected, and it's why T5
re-measures the baseline.

---

## T1 · Verification harness and CI

**Status:** done (PR #2)

Set up the script-driven harness that every later task relies on.

- Add `pytest` as a dev dependency.
- Add a `tests/` package with fixtures for a tiny model config and **synthetic** token JSON
  files written to a temp dir. Tests and CI must not depend on `data/`, which is gitignored.
- Write baseline tests that pass on today's code:
  - parameter count matches the formula in `docs/architecture/model-and-training.md`
  - every parameter receives a gradient (the regression test for the old Q=K=V bug)
  - causality: changing token t leaves logits before t unchanged
  - the model can overfit one batch to near-zero loss
  - `RandomGuitarSeqDataset` returns in-bounds, shifted `(x, y)` windows
- Write tests for known bugs as `xfail(strict=True)`, each naming its issue number:
  - short-prompt forward (#1)
  - `top_k` (#2)
  - compiled-checkpoint load (#3)

  `strict=True` makes each one fail loudly once its bug is fixed, so the marker has to be
  removed.
- Scripts:
  - `scripts/verify.sh`: pytest. The smoke train runs inside pytest
    (`tests/test_smoke_train.py`), so its checkpoint can also back the #3 test.
  - `scripts/smoke-train.sh`: a 1-epoch tiny run on synthetic data, or on `--data_root` if
    given
  - `scripts/setup.sh`: `uv sync`
- CI: a GitHub Actions workflow that runs `scripts/setup.sh` and `scripts/verify.sh` on
  ubuntu CPU. Use uv's CPU-only torch wheel in CI if the default CUDA wheel makes the job
  too slow.
- Update `AGENTS.md`: replace "Until T1 lands" with the verify gate.

**Acceptance:**

- [x] `scripts/verify.sh` exits 0 on a fresh clone after `scripts/setup.sh`, in under ~2 min
      locally. (~8 s on MPS.)
- [x] CI runs the same script and is green. (~80 s, of which `uv sync` is 27 s with the
      default CUDA torch wheel, so no CPU-only index was needed.)
- [x] The xfail tests for #1–#3 fail as expected today.

## T2 · Model correctness and sampling

**Status:** todo

Make the model usable for generation.

- #1: slice `rel_bias.weight[-T:]` so that any `T ≤ block_size` works. Parameter shapes don't
  change, so existing checkpoints still load.
- #2: implement `top_k` in `generate()`, and optionally `top_p`. `generate()` should switch
  to eval mode and `no_grad` itself, then restore the previous mode.
- #21: tidy `model.py`:
  - remove the unused hyperparameters import
  - fix the stale "scalar relative bias" comment
  - remove the unused `FeedForward.dropout`
  - add an `n_embd % n_head` assert
  - keep `Head` for now; Phase 2 replaces attention
- #25: update `playground/model_run.ipynb` so it loads a checkpoint and generates from a short
  prompt, with the stale `rel_bias` plot removed.

**Acceptance:**

- [ ] The #1 and #2 xfails are removed and those tests pass.
- [ ] `generate()` from a 16-token prompt produces `max_new_tokens` tokens with
      `top_k=40, temperature=0.9`.
- [ ] `scripts/verify.sh` is green.

## T3 · Training loop and checkpoints

**Status:** todo

Make runs resumable and correct.

- #3: save the uncompiled `state_dict` (unwrap `_orig_mod`), and strip the prefix when loading
  older checkpoints.
- #4: save and restore the scheduler state, so a resumed run continues its LR schedule
  instead of restarting warmup.
- #5: one `save_checkpoint` helper with a single schema for `best.pt`, `epoch_N.pt` and
  `final.pt`.
- #6: scale the last partial accumulation group by its real size.
- #7: fix the dataset window off-by-ones.
- #13, #20: remove the duplicate `grad_accum` calculation and the unused folder paths.
- #18: make paths work from the repo root. `--data_root` defaults to `data`, and
  `--checkpoint_dir` defaults to `{data_root}/checkpoints`.
- #24: device-generic autocast and GradScaler, and accurate log lines.

**Acceptance:**

- [ ] Test: train 2 steps, save, resume. The optimizer step count, scheduler LR and epoch
      continue from where they stopped.
- [ ] Test: a checkpoint from a compiled model loads into an uncompiled model.
- [ ] The #3 xfail is removed.
- [ ] The commands in `AGENTS.md` work without the `--checkpoint_dir` workaround. Update the
      docs.
- [ ] `scripts/verify.sh` is green.

## T4 · Run configs, metrics and a fixed val set

**Status:** todo

Make each run reproducible and comparable.

- **Run spec:** a TOML file read with stdlib `tomllib`, with CLI flags overriding its values.
  Add `configs/smoke.toml` and `configs/colab-186m-jan2026.toml`; the latter reproduces the
  2026-01-31 run.
- **Run directory** per run, `{data_root}/runs/<timestamp>-<name>/`, containing the resolved
  config, `metrics.jsonl` and the checkpoints.
- **Metrics per epoch:** train/val loss, ppl, LR, grad norm, tokens/s, and MFU
  (`6·N·tokens/s ÷ device peak FLOPs`).
- #10: a deterministic val set (fixed windows), so that val loss is comparable across epochs
  and runs.
- `scripts/train.sh <config> [overrides]`.
- Keep `hyperparameters.py` as the CLI defaults for now. Phase 2 can fold it into configs.

**Acceptance:**

- [ ] `scripts/train.sh configs/smoke.toml` writes a run directory with `config.json` and
      `metrics.jsonl`.
- [ ] Two runs with the same config and seed give the same val loss on the fixed val set
      (CPU).
- [ ] `docs/architecture/model-and-training.md` is updated.
- [ ] `scripts/verify.sh` is green.

## T5 · Scripted Colab run and baseline

**Status:** todo

Test `docs/colab.md` end to end, and measure where compute goes.

- Script the Colab workflow under `scripts/colab/`: provision, sync code, launch detached,
  poll, fetch artifacts, stop. Correct `docs/colab.md` wherever reality differs.
  `colab drivemount` stays a manual step for the owner.
- Run `configs/colab-186m-jan2026.toml` on an A100 for a few epochs. Record tokens/s, MFU, and
  a `torch.profiler` summary of a few steps showing the share of time in attention vs the
  rest.
- Run it to completion for the **post-fix baseline** val loss on the fixed val set. It
  replaces 3.46 as the number Phase 2 compares against.

**Acceptance:**

- [ ] Every step except drivemount runs from a script.
- [ ] A journal entry `docs/journal/<date>.md` records MFU, tokens/s, the profile summary and
      the baseline val loss.
- [ ] `docs/architecture/scaling.md` run history is updated.

---

**Story done when:** T1–T5 are done, `scripts/verify.sh` is green in CI, and the issues fixed
here are marked fixed in `docs/known-issues.md`.
