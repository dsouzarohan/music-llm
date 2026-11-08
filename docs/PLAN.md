# Plan: Phase 2, Modern block and small rebaseline

**Story status: in progress** (started 2026-11-08). Branch `feat/phase-2-modern-block`, shipped as
one PR. Phase 1's plan is in git history (PRs #2–#4).

**Story:** bring the model up to current practice and find out whether model size or data is
the bottleneck. After this phase:

- runs happen on Lightning.ai over SSH, from one script, into a tidy run layout;
- attention is one batched SDPA call, with no per-head loop and no mask buffers in checkpoints;
- the block is chosen by measurement (ablations at ~30M), not by habit;
- sampling is fast enough to listen to many samples (KV cache);
- there is a measured answer to "small model, more steps vs big model" at equal compute.

**Status legend:** `todo` · `in progress` · `done`. Issue numbers (#n) refer to
`docs/known-issues.md`.

**Baseline to beat:** val **3.498** (ppl 33.1) on the fixed val set, 186M params, 204.8M tokens,
≈3.7e17 training FLOPs, 22.2k tok/s, MFU 12.8% on an A100 40GB (`docs/journal/2026-11-08.md`).

**Out of scope for this phase:**

- Data pipeline changes and the corpus itself (#8, #9, #11–#17, #19, `scripts/stats.sh`):
  Phase 3. This phase trains on the existing Drive corpus (vocab 3500, ~49M train tokens).
- Tokenizer changes and per-attribute RoPE: Phase 4.
- Custom kernels: Phase 7. SDPA's built-in kernels are enough here.

**Checkpoints:** T3 changes parameter names and shapes, so **every checkpoint from before T3,
including the baseline `best.pt`, stops loading.** The baseline's numbers stay recorded; its
weights are not reused.

---

## Decisions

Made while planning (2026-11-08). Revisit here, not silently in code.

1. **Lightning and Colab share one interface.** `scripts/remote.sh` has the same verbs as
   `scripts/colab.sh` (setup, train, logs, status, fetch, stop), over plain `ssh`/`rsync`.
   Colab stays usable while its credits last.
2. **The corpus is copied from Drive to the Studio's persistent disk once** (rclone with a
   Drive remote, or whatever Lightning offers; confirmed in T1). Drive stays the canonical copy.
3. **Run layout:** `data/runs/<YYYY-MM-DD>/<HHMM>-<name>/` with `config.json`, `metrics.jsonl`
   and `checkpoints/{best.pt, last.pt, epoch_NNN.pt}`. Fixed checkpoint names, no timestamps in
   filenames. `last.pt` is what resume uses, `best.pt` is what sampling uses. Old runs keep
   their old layout.
4. **Huang relative attention survives as an option**, rewritten as an additive bias into
   `scaled_dot_product_attention` with all heads batched. It's the control in every ablation.
   A test proves the rewrite matches the current per-head code before the old code is deleted.
5. **Each modern piece is a config switch** (`pos`, `norm`, `mlp`, `qk_norm`, `tie_embeddings`),
   so the T5 ablations are config files, not branches. The defaults flip to the winning
   combination at the end of T5.
6. **Block choice at ~30M, scale question afterwards.** Comparing a new-block 30M model with the
   old-block 186M baseline changes two things at once. T5 isolates the block, T6 isolates size.
7. **Context stays 2048, vocab stays 3500.** Only the block changes.

---

## T1 · Run layout and checkpoint retention

**Status:** done

A chore, done first so every run in this phase uses it.

- `resolve_run_dir` → `{runs_dir}/<YYYY-MM-DD>/<HHMM>-<name>/`; checkpoints go in a
  `checkpoints/` subfolder.
- `best.pt` (overwritten when val improves), `last.pt` (overwritten every `save_every` and at
  the end; replaces `final.pt`), `epoch_NNN.pt` snapshots (zero-padded).
- `--keep_last N` (default 2): delete older `epoch_NNN.pt` after each save. `best.pt` and
  `last.pt` are never deleted.
- Atomic saves: write `<name>.tmp`, then `os.replace` it over the target.
- `--resume` accepts a run directory (uses `checkpoints/last.pt`) or a checkpoint file. Resume
  continues in the same run directory.
- Update `scripts/colab.sh` and `docs/colab.md` paths, `sample.py` docstring, `AGENTS.md`.

**Acceptance:**

- [x] Test: a 3-epoch run with `save_every=1, keep_last=2` leaves `best.pt`, `last.pt` and
      exactly the 2 newest snapshots.
- [x] Test: `--resume <run_dir>` continues exactly (the existing resume test, via the directory).
- [x] Test: an interrupted save (exception mid-write) leaves the previous `last.pt` loadable.
- [x] `scripts/verify.sh` is green.

## T2 · Lightning.ai setup

**Status:** todo

- Owner: create a Studio, add the SSH key, note the host alias. Pick the GPU per run
  (L40S / A100 / H100); MFU uses the detected GPU's peak.
- `scripts/remote.sh` (env: `HOST`, `BRANCH`): `setup` clones or resets the repo and runs
  `scripts/setup.sh`; `train <config> [flags]` launches detached with `nohup python -u` and
  prints the run dir; `logs`, `status`, `fetch <run>` (metrics, config, samples; checkpoints only
  on request), `stop` (or how to stop the Studio, if that's only in the UI).
- One-time corpus copy from Drive to persistent disk; `remote.sh setup` checks it's there
  (file counts: 17,233 train, 3,365 val).
- `docs/lightning.md`: setup, the workflow, costs, and gotchas as we find them.
- Add `PEAK_FLOPS` entries for any new GPU type.

**Acceptance:**

- [ ] From a fresh Studio, `remote.sh setup` then `remote.sh train configs/smoke.toml
      --data_root <corpus>` completes, and `remote.sh fetch` brings back its metrics.
- [ ] A few epochs of `configs/colab-186m-jan2026.toml` reproduce the baseline's early val
      losses (7.73 at epoch 0) and record tokens/s and MFU on the Lightning GPU.
- [ ] `docs/lightning.md` written; `AGENTS.md` "Where things run" updated.

## T3 · Batched attention and the modern block

**Status:** todo

Rewrite attention, then add the modern pieces behind switches.

- **Attention:** one `qkv` projection, `(B, H, T, D)` tensors, one
  `F.scaled_dot_product_attention(..., is_causal=True)` call. Fixes #23 and #26 (no `tril`
  buffers).
- **`pos = "huang"`**: the relative bias becomes a `(H, T, T)` additive `attn_mask` built from
  `rel_bias`, plus the absolute position embedding, as today. Equivalence test against the
  current `RelativeHead` code with copied weights, then delete `Head`/`RelativeHead`.
- **`pos = "rope"`**: rotary embeddings on q and k; no absolute position embedding, no
  `rel_bias`. SDPA can use its flash kernel because there's no bias.
- **`norm = "layernorm" | "rmsnorm"`**, **`qk_norm`**: RMSNorm on q and k per head.
- **`mlp = "gelu" | "swiglu"`**: SwiGLU hidden size `≈ 8/3·C` rounded, to keep params
  comparable; also removes the hardcoded `mlp_ratio` (#21 leftover).
- **`tie_embeddings`**: share the token embedding and `lm_head` weights.
- **Weight-decay groups (#22):** decay 2-D weights only; no decay on norms, biases, embeddings.
- Update the parameter-count formula in `docs/architecture/model-and-training.md` and its test
  for each switch; update the 186M meta-model test.

**Acceptance:**

- [ ] Batched Huang attention matches the old per-head implementation's logits (copied weights,
      fp32, `atol` ~1e-5).
- [ ] Existing model tests (causality, all params get grads, overfit, short prompt, prefix
      logits) pass for every switch combination.
- [ ] A 186M checkpoint is ≈2.2 GB (no mask buffers).
- [ ] Eager profile at micro-batch 2 shows attention's share of device time well below 38%;
      tokens/s and MFU recorded for both `pos` modes.
- [ ] `scripts/verify.sh` is green.

## T4 · KV cache

**Status:** todo

- Per-layer K/V cache in `generate()`: process the prompt once, then one token per step.
- Works for both `pos` modes (Huang bias for one query row; RoPE at the token's position).
- Past `block_size`: drop the cache and re-prefill from the last `block_size − 1` tokens.
- `sample.py` reports seconds per sample and uses the cache by default (`--no-cache` to compare).

**Acceptance:**

- [ ] Test: cached and uncached greedy decoding produce identical tokens, both `pos` modes,
      including across the `block_size` boundary.
- [ ] A 512-token sample from a 186M-shape model takes seconds instead of ~108 s on the
      Lightning GPU (recorded in the journal).
- [ ] `scripts/verify.sh` is green.

## T5 · Block ablations at ~30M

**Status:** todo

- `configs/lightning-30m-*.toml`: ~30M params (e.g. 8 layers, 512 dim, 8 heads), context 2048,
  the same token budget for every run (≈200M tokens, confirmed once tokens/s is known).
- Runs, all on the fixed val set:
  1. old block (`huang`, layernorm, gelu, untied)
  2. + RoPE
  3. + RMSNorm and QK-norm
  4. + SwiGLU
  5. + tied embeddings (full modern block)
- Pick the block from val loss, then tokens/s. Flip the defaults in `hyperparameters.py` to it.

**Acceptance:**

- [ ] All five runs logged in a journal entry with val loss, ppl, tokens/s and MFU.
- [ ] The chosen block is the default, with the decision and its numbers in the journal.

## T6 · Small vs big at equal compute

**Status:** todo

- Train the chosen ~30M block on the baseline's compute budget (≈3.7e17 FLOPs by
  `metrics.flops_per_token`, ≈1.2B tokens, ~24 passes over the corpus).
- Compare with 3.498. Watch the train/val gap: repeating the data ~24× may overfit, which is
  itself the answer to "is data the bottleneck".
- Sample with `sample.py` (KV cache) and record grammar error rates.

**Acceptance:**

- [ ] A journal entry with the val curve, the comparison with the baseline at equal FLOPs, and
      the overfitting picture.
- [ ] `docs/architecture/scaling.md` run history and `docs/ROADMAP.md` (Phase 2 done, Phase 3
      inputs) updated.

---

**Story done when:** T1–T6 are done, `scripts/verify.sh` is green in CI, #22, #23 and #26 are
marked fixed, and the docs describe the new block and the Lightning workflow.
