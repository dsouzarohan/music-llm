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
≈3.16e17 training FLOPs, 22.2k tok/s, MFU 11.0% on an A100 40GB (`docs/journal/2026-11-08.md`;
that entry says 3.7e17 and 12.8%, from the FLOPs formula before Decision 10).

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
8. **T3 and T4 go before T2** (2026-11-08, Lightning outage). Their code and CPU tests don't need
   a cloud GPU. Their GPU acceptance items (T3 profile and tokens/s, T4 sample timing) stay open
   until T2 lands, then get measured together.
9. **T5 runs share one shape, not one parameter count** (2026-11-08). At 8L/512/8H the old block
   is 34.0M params and the full modern block 22.8M: Huang adds 8.4M of `rel_bias` plus a 1.0M
   position table, and tying removes the 1.8M `lm_head`. All five runs keep the same layers, width
   and heads, so the weights that model content are the same size; the journal reports total
   params and FLOPs/token for each run.
10. **FLOPs/token counts matmuls only** (2026-11-08, T3). `6 × nn.Linear weights + 6·T·C per
    attention matmul per layer` (2 matmuls, 3 with Huang's `q@Eᵀ`). The old `6N + 18·L·C·T`
    charged FLOPs to embedding lookups and to `rel_bias`, which the attention term already
    counted, so it overstated the baseline by 14% (1.798 → 1.545 GFLOP/token).

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

**Status:** in progress (code done; the GPU profile waits for T2)

Rewrite attention, then add the modern pieces behind switches.

- **Attention:** one `qkv` projection, `(B, H, T, D)` tensors, one
  `F.scaled_dot_product_attention(..., is_causal=True)` call. Fixes #23 and #26 (no `tril`
  buffers).
- **`pos = "huang"`**: the relative bias becomes a `(B, H, T, T)` additive `attn_mask` (it
  depends on q, so it's per batch element) built from `rel_bias` and the causal mask, plus the absolute position embedding, as today. Equivalence test against the
  current `RelativeHead` code with copied weights, then delete `Head`/`RelativeHead`.
- **`pos = "rope"`**: rotary embeddings on q and k; no absolute position embedding, no
  `rel_bias`. SDPA can use its flash kernel because there's no bias.
- **`norm = "layernorm" | "rmsnorm"`**, **`qk_norm`**: RMSNorm on q and k per head.
- **`mlp = "gelu" | "swiglu"`**: SwiGLU hidden size `2/3 · mlp_ratio · C` rounded up to 64
  (`2C` at our ratio 3; the usual `8/3·C` assumes ratio 4), to keep params equal;
  `mlp_ratio` becomes a hyperparameter (#21 leftover).
- **FLOPs/token** from the model's matmuls (Decision 10).
- **`tie_embeddings`**: share the token embedding and `lm_head` weights.
- **Weight-decay groups (#22):** decay 2-D weights only; no decay on norms, biases, embeddings.
- Update the parameter-count formula in `docs/architecture/model-and-training.md` and its test
  for each switch; update the 186M meta-model test.

**Acceptance:**

- [x] Batched Huang attention matches the old per-head implementation's logits (copied weights,
      fp32, `atol` ~1e-5).
- [x] Existing model tests (causality, all params get grads, overfit, short prompt, prefix
      logits) pass for every switch combination. (Six variants, `tests/conftest.py::BLOCKS`:
      the T5 ladder plus the modern parts on Huang. Each switch is tested on and off, and each
      `pos` mode both with and without the rest.)
- [x] A 186M checkpoint is ≈2.2 GB (no mask buffers): 186.4M fp32 weights (0.75 GB) + AdamW's
      two moments (1.49 GB) = 2.24 GB; `test_checkpoint_has_no_buffers` checks the state dict
      holds parameters only.
- [ ] Eager profile at micro-batch 2 shows attention's share of device time well below 38%;
      tokens/s and MFU recorded for both `pos` modes.
- [x] `scripts/verify.sh` is green.

## T4 · KV cache

**Status:** in progress (code done; the GPU timing waits for T2)

- Per-layer K/V cache in `generate()`: process the prompt once, then one token per step.
- Works for both `pos` modes (Huang bias for one query row; RoPE at the token's position).
- Past `block_size`: drop the cache and re-prefill from the last `block_size` tokens. That's
  the same window the uncached path sees, so outputs match exactly, but every step past the
  boundary costs a full forward pass. Keeping a shorter window to amortise it is left for later.
- `sample.py` reports seconds per sample and uses the cache by default (`--no-cache` to compare).

**Acceptance:**

- [x] Test: cached and uncached greedy decoding produce identical tokens, both `pos` modes,
      including across the `block_size` boundary. (`test_cached_and_uncached_greedy_decoding_match`
      for all six block variants, plus per-step logits vs a full forward.)
- [ ] A 512-token sample from a 186M-shape model takes seconds instead of ~108 s on the
      Lightning GPU (recorded in the journal). Local reference, random-init 186M shape on an
      M-series Mac (MPS, fp32), 64-token prompt + 256 tokens: Huang 8.5 → 2.6 s, RoPE 6.1 → 2.2 s.
- [x] `scripts/verify.sh` is green.

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
- **Probe (after the runs):** for the RoPE runs, split each head's q and k energy by RoPE
  frequency band (fast pairs vs slow pairs), per token type (Bar, Position, Pitch, Velocity,
  Duration, and BPE merges by their first base token). Shows whether heads use fast pairs
  for local, positional lookups and slow pairs for content matching, and whether that differs
  by token type. A notebook in `playground/` plus a journal note.

**Acceptance:**

- [ ] All five runs logged in a journal entry with val loss, ppl, tokens/s and MFU.
- [ ] The chosen block is the default, with the decision and its numbers in the journal.
- [ ] Frequency-band probe results in the journal.

## T6 · Small vs big at equal compute

**Status:** todo

- Train the chosen ~30M block on the baseline's compute budget (≈3.16e17 FLOPs by
  `metrics.flops_per_token`; ≈1.3B tokens for the full modern block at 237 MFLOP/token, ~27
  passes over the corpus).
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
