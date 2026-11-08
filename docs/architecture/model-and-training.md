# Model & training architecture

Code: `src/model/model.py`, `src/utils/train.py`, `src/utils/hyperparameters.py`.

## Model at a glance

A GPT-2-shaped pre-norm decoder. The block's parts are switches (`MusicTransformer` kwargs, CLI
flags and TOML keys of the same name; defaults in `hyperparameters.py`):

| Switch | Values | Default |
|---|---|---|
| `pos` | `huang`: Music Transformer relative attention + absolute position embedding · `rope`: rotary embeddings on q and k, no position embedding | `huang` |
| `norm` | `layernorm` · `rmsnorm` (block norms and `ln_f`) | `layernorm` |
| `mlp` | `gelu`: hidden `mlp_ratio·C` · `swiglu`: hidden `2/3·mlp_ratio·C` rounded up to 64, the same parameter count | `gelu` |
| `mlp_ratio` | GELU hidden size multiple | 3 |
| `qk_norm` | RMSNorm (over Dh, weights shared by heads) on q and k | off |
| `tie_embeddings` | `lm_head.weight` is the token embedding tensor | off |

The defaults are the pre-Phase-2 block. Phase 2 T5 picks the new defaults by ablation.

```
idx (B,T)
 ├─ token_embeddings  Embedding(V, C)
 └─ position_embeddings Embedding(block_size, C)     ← pos="huang" only
x = tok (+ pos)
for each of n_layer Blocks:
    x = x + Attention(norm1(x))                      ← qkv Linear(C,3C) → SDPA (all heads) → proj Linear(C,C) → dropout
    x = x + FeedForward(norm2(x))                    ← up → GELU or SwiGLU → down → dropout
x = norm_f(x)
logits = last Linear(C, V)                           ← tied to token_embeddings if tie_embeddings
loss = cross_entropy(logits, targets)                ← no ignore_index (there is no padding)
```

### `Attention`

One `qkv = Linear(C, 3C, bias=False)` whose output rows are `[q heads | k heads | v heads]`.
It's reshaped to q, k, v of shape (B, H, T, Dh), with `Dh = C // n_head`. With `qk_norm`, q and
k each go through an RMSNorm over Dh. Then all heads run in one
`F.scaled_dot_product_attention` call (SDPA). Heads are concatenated back to (B, T, C) and go
through `proj` and dropout. There are no buffers in the state dict: the causal mask is built per
call, and the RoPE tables are non-persistent buffers.

**`pos="huang"`** (Huang et al., 2018, with the skewing trick). Each layer has
`rel_bias`, an `(H, block_size, Dh)` parameter: one learned vector per head per relative
distance. Row `block_size-1` is distance 0 and row 0 is distance `-(block_size-1)`.
`Attention.huang_mask(q)`:

1. `E = rel_bias[:, -T:]`, the last T rows, covering distances `-(T-1) … 0`: all a length-T
   sequence can use. `S_rel = q @ Eᵀ` has shape (B, H, T, T). Row *i* holds q_i's affinity to
   every relative distance.
2. **Skew.** Pad one zero column on the left, reshape to (B, H, T+1, T), and drop the first
   row. This shifts row *i* left by (T-1-i), which turns "query × distance" into
   "query × key-position" without building the (T, T, Dh) tensor that Shaw et al. need.
3. Scale by `Dh^-0.5` (SDPA scales `q@kᵀ` by the same factor, so both terms are scaled
   together; the 2025-12-04 fix) and fill above the diagonal with `-inf` (causal).

The result goes to SDPA as `attn_mask`, which adds it to `q@kᵀ·Dh^-0.5` before the softmax. The
mask depends on q, so it's (B, H, T, T), and SDPA can't use its flash kernel.
`tests/test_attention.py` checks this path against a frozen copy of the pre-Phase-2 per-head
code (`tests/reference_attention.py`).

**`pos="rope"`** (Su et al., 2021). Channel pairs (i, i + Dh/2) of q and k are rotated by
`position · θ_i`, with `θ_i = 10000^(-2i/Dh)`. A rotation by m on q and by n on k leaves q·k
depending only on m − n, so scores are relative while each token is rotated by its own index.
Rotations keep norms, so the `Dh^-0.5` scaling is unchanged. `rope_tables` holds cos/sin for
every position up to `block_size`, cast to q's dtype before use. SDPA then runs with
`is_causal=True` and no mask, so the flash kernel is available. The fastest pair repeats every
6.3 tokens and the slowest (Dh=64) every ~47k tokens.

Any `T ≤ block_size` works in both modes. Because attention is causal and position enters only
as distance (Huang) or as each token's own index (RoPE), running the first T tokens alone gives
the same logits as positions `0…T-1` of a full window; `test_prefix_logits_match_full_sequence`
checks this.

### Initialisation (GPT-2 style, `MusicTransformer.__init__` and `_init_weights`)

- All Linear and Embedding weights are drawn from N(0, 0.02), and all biases are set to 0.
  `rel_bias` is drawn from N(0, 0.02). Norm weights start at 1.
- Residual output projections (`attn.proj.weight` and `ffwd.down.weight`) are re-drawn with std
  `0.02 / sqrt(2 · n_layer)`, so the residual stream's variance doesn't grow with depth.
- With `tie_embeddings`, `lm_head` takes the token embedding's tensor after init.

## Parameter count

`tests/test_model.py::expected_params` checks this formula for every tested block variant.
With the default block, the 24-layer config gives exactly the **186,426,284** printed in the
2025-12-29 Colab log.

With C = embedding dim, V = vocab, T = block_size, L = layers, H = heads, Dh = C/H, r = `mlp_ratio`:

```
norm        = 2C (layernorm) or C (rmsnorm)
attention   = 3C² (qkv) + (C² + C) proj + [huang: T·C rel_bias] + [qk_norm: 2·Dh]
ffn         = gelu:   h = r·C,                     (C·h + h) + (h·C + C)
              swiglu: h = 64·ceil(2rC / (3·64)),   (C·2h + 2h) + (h·C + C)
per block   = attention + ffn + 2·norm
total       = V·C + [huang: T·C pos emb] + L·per_block + norm (final) + [untied: C·V] + V (lm_head bias)
```

| Config | V | C | L | H | T | Params | of which `rel_bias` |
|---|---|---|---|---|---|---|---|
| `hyperparameters.py` defaults | 3000 | 384 | 8 | 4 | 512 | 15.9M | 1.6M |
| 2025-12-03/27 Colab runs | 3500 | 768 | 12 | 12 | 2048 | ~97M | 18.9M |
| 2025-12-29 and 2026-01-31 runs | 3500 | 768 | 24 | 12 | 2048 | **186.4M** | **37.7M (20%)** |
| Phase 2 ~30M shape, old block | 3500 | 512 | 8 | 8 | 2048 | 34.0M | 8.4M (25%) |
| Phase 2 ~30M shape, modern block (rope, rmsnorm, swiglu, qk_norm, tied) | 3500 | 512 | 8 | 8 | 2048 | 22.8M | 0 |

The `rel_bias` tables grow with `block_size × C × L`. They are the reason a 2048 context costs
20% of the model with `pos="huang"`.

## Training loop (`train.py`)

| Aspect | What it does |
|---|---|
| Entry | `scripts/train.sh <config.toml> [flags]`, i.e. `python -m src.utils.train --config …` from the repo root. `train(args)` is also callable in-process (tests do this). |
| Config | A flat TOML run spec in `configs/` whose keys are the CLI options. Precedence: CLI flag > TOML > `hyperparameters.py` defaults. Unknown TOML keys are an error. |
| Run directory | `{runs_dir}/<YYYY-MM-DD>/<HHMM>-<name>/`, `runs_dir` defaulting to `{data_root}/runs` (suffixed `-2`, `-3`, … if taken; or `--run_dir`; a resume writes into the resumed run's directory). Holds `config.json` (resolved args plus device, param count, FLOPs/token, git commit), `metrics.jsonl` and `checkpoints/`. Runs before Phase 2 T1 use the old flat layout (`<YYYYMMDD-HHMMSS>-<name>/` with checkpoints beside `config.json`). |
| Data | `{data_root}/tokenized/train-aug/*.json` and `{data_root}/tokenized/val/*.json` |
| Train set | `RandomGuitarSeqDataset`: `--train_windows` (default 2000) random windows per epoch, each from a random file weighted by its number of valid starts. `--use_guitar_ds` switches to sliding windows. |
| Val set | **Fixed**: `GuitarDataset` with stride `block_size` (non-overlapping windows over every val file), thinned to `--val_windows` (default 512) evenly spaced windows. The same files always give the same windows, so val loss is comparable across epochs and runs. |
| Batching | `--batch_size` is the effective batch and `--micro_batch_size` is what fits in memory. `grad_accum = batch // micro`. Each micro-batch loss is divided by the size of its accumulation group, so a short last group isn't under-weighted. |
| Optimizer | AdamW(lr, betas=(0.9, 0.95)) over `model.param_groups(--weight_decay)`: decay (default 0.01) on `nn.Linear` weights only; none on embeddings (including a tied `lm_head`), `rel_bias`, norms and biases. |
| Schedule | `OneCycleLR` over all `epochs`, stepped per optimizer step: 10% warmup from lr/25, then cosine down to lr/25000 |
| Clipping | `clip_grad_norm_(…, --grad_clip)`, default 0.5 |
| Precision | CUDA only: bf16 autocast if supported, else fp16 + GradScaler. TF32 on. MPS/CPU run fp32. |
| Compile | `torch.compile(model)` unless `--no-compile` |
| Device | `--device auto`: cuda → mps → cpu |
| Metrics | One `metrics.jsonl` row per epoch: `train_loss`, `val_loss`, `val_ppl`, `val_bits_per_token`, `lr`, mean `grad_norm` (before clipping), `tokens_per_sec`, `mfu`, `train_time_s`. |
| MFU | `tokens/s × FLOPs/token ÷ peak FLOP/s`. `metrics.flops_per_token(model)` = 6 per `nn.Linear` weight (2 forward + 4 backward; a tied `lm_head` counts once) + `6·L·C·T` per T×C attention matmul per token: `q@kᵀ` and `attn@v`, plus `q@Eᵀ` with `pos="huang"`. Embedding lookups, `rel_bias`, norms and biases add no matmul FLOPs and aren't counted. (Before Phase 2 it was `6N + 18·L·C·T` over all parameters, 14% higher for the 186M model.) Peak is looked up for H100/A100/L4/T4 (dense bf16) or set with `--peak_flops`; otherwise `mfu` is null. |
| Profiling | `--profile_steps N` profiles N micro-steps of the first epoch into `{run_dir}/profile.txt`: forward time in the `attention` and `mlp` regions (`Block.forward` labels) as a share of step time, then the op table. The labels disappear under `torch.compile`, so profile the split with `--no-compile`. |
| Checkpoints | In `checkpoints/`: `best.pt` (on val improvement), `last.pt` (every `--save_every` epochs and at the end), `epoch_NNN.pt` snapshots (every `--save_every` epochs; only the `--keep_last` newest are kept, default 2, 0 writes none). Each save writes `<name>.tmp`, fsyncs it, then `os.replace`s it over the target, so an interrupted save leaves the previous file intact. All use one schema: `model` (uncompiled keys), `optimizer`, `scheduler`, `scaler`, `epoch`, `step`, `val_loss`, `best_val_loss`, `vocab_size`, `hyperparams`, `config`, `rng`. `load_checkpoint` strips the `_orig_mod.` prefix from older checkpoints. |
| Resume | `--resume <run_dir>` (uses `checkpoints/last.pt`) or `--resume path.pt` restores model, optimizer, scheduler, scaler, step, epoch and RNG state, so the run continues exactly: same LR schedule, same data order, same dropout masks. Pre-T3 checkpoints have no scheduler state; the schedule then restarts over the remaining epochs. |
| Fine-tune | `--finetune_from path.pt` loads model weights only, with a fresh optimizer and schedule. |

One epoch here means `epoch_len / micro_batch` micro-steps, **not** one pass over the corpus.
For the current Colab config: 2000 windows / 8 = 250 micro-steps, which at 16× accumulation
is ~16 optimizer steps and ~4.2M tokens. So 50 epochs is 800 optimizer steps and ~210M tokens
seen.

## Inference

`MusicTransformer.generate(idx, max_new_tokens, temperature, top_k, top_p)` samples one
token at a time from `logits[:, -1]`, feeding the model the last `block_size` tokens. Any
prompt length works.

- `temperature` divides the logits: below 1 is more conservative, above 1 more random.
- `top_k` keeps the k most likely tokens.
- `top_p` (nucleus) keeps the smallest set of most likely tokens whose probability reaches
  `top_p`.
- It switches to eval mode and `no_grad` itself, and restores the previous mode afterwards.
- There is no EOS stop and no KV cache: every step reruns the whole window.

`playground/model_run.ipynb` loads `data/checkpoints/best.pt` (or `$MUSIC_LLM_CKPT`) with
`load_checkpoint`, which strips the `_orig_mod.` prefix pre-T3 compiled runs saved, and rebuilds the model
from `ckpt["hyperparams"]`. It prompts with the first 64 tokens of a val song, generates 1024
tokens with `temperature=0.9, top_k=40, top_p=0.95`, and decodes both the prompt and the
output to `data/output/*.mid` with `REMI(params=data/tokenized/config/tokenizer.json)`. It
also prints the **grammar error rate**: the share of generated tokens that break REMI's
`Program → Pitch → Velocity → Duration` note structure. miditok drops broken notes when
decoding, so an undertrained model can produce MIDI with no new notes at all.

## How the architecture got here (from the journal)

| Date | Change | Why |
|---|---|---|
| 2025-09-10 | nanoGPT-style transformer written from scratch | baseline |
| 2025-09-13 | fixed `** Dh**-0.5` → `* Dh**-0.5` | all-NaN attention |
| 2025-10-26 | SDPA in `Head`, bf16 | memory on 5070 Ti |
| 2025-11-15 → 11-20 | relative attention: first a scalar bias per distance, then Huang skewing | music is about relative timing and intervals |
| 2025-12-01/03 | all hparams exposed as CLI args, gradient accumulation | fit 2048 context on an A100 |
| 2025-12-04 | OneCycleLR, batch 64→128, max lr 1e-4, clip 1.0→0.5 | gradient explosions around epoch 4 and epoch 24 |
| 2025-12-04 | scale `(content + rel)` together instead of content only | the unscaled rel term was 8× too strong and made the model position-dominated |
| 2025-12-29 | GELU, GPT-2 init, 12→24 layers | depth without explosion |
| 2026-01-31 | **fixed `k = v = self.query(x)`** (commit 997d968) | every relative-attention run before this had broken attention |
| 2026-11-08 | Phase 2 T3: batched SDPA attention, block switches (`pos`, `norm`, `mlp`, `qk_norm`, `tie_embeddings`), weight-decay groups, matmul-only FLOPs | per-head loop and mask buffers cost speed and checkpoint size; T5 ablates the modern block |

Treat results from before 2026-01-31 as "trained with Q=K=V". They are useful for the stability
lessons, but not as architecture baselines.
