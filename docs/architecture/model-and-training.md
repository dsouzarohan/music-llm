# Model & training architecture

Code: `src/model/model.py`, `src/utils/train.py`, `src/utils/hyperparameters.py`.

## Model at a glance

A GPT-2-shaped decoder (pre-LayerNorm). Its self-attention uses Music Transformer relative
attention (Huang et al., 2018), implemented with the memory-efficient "skewing" trick.

```
idx (B,T)
 ├─ token_embeddings  Embedding(V, C)
 └─ position_embeddings Embedding(block_size, C)     ← absolute positions, kept alongside relative attn
x = tok + pos
for each of n_layer Blocks:
    x = x + MultiHeadAttention(LN1(x))               ← n_head × RelativeHead, concat, proj Linear(C,C), dropout
    x = x + FeedForward(LN2(x))                      ← Linear(C,3C) → GELU → Linear(3C,C) → dropout
x = LN_f(x)
logits = last Linear(C, V)                           ← not weight-tied to token_embeddings
loss = cross_entropy(logits, targets)                ← no ignore_index (there is no padding)
```

### `RelativeHead` (the attention actually used)

Each head owns these parameters:

- `query`, `key`, `value` are `Linear(C, Dh, bias=False)`, with `Dh = C // n_head`.
- `rel_bias` is `Embedding(block_size, Dh)`: one learned vector per relative distance.
  Row `block_size-1` is distance 0 and row 0 is distance `-(block_size-1)`.

The forward pass (`RelativeHead.forward`):

1. `q, k, v` are projected from `x`.
2. `E = rel_bias.weight[-T:]`, the last T rows, covering distances `-(T-1) … 0`: all a
   length-T sequence can use. `S_rel = q @ Eᵀ` has shape (B, T, T). Row *i* holds q_i's
   affinity to every relative distance.
3. **Skew.** Pad one zero column on the left, reshape to (B, T+1, T), and drop the first row.
   This shifts row *i* left by (T-1-i), which turns "query × distance" into
   "query × key-position" without building the (T, T, Dh) tensor that Shaw et al. need.
   That is the memory trick from the Music Transformer paper.
4. Compute `scores = (q @ kᵀ + S_rel) * Dh^-0.5`. Both terms are scaled together; the
   2025-12-04 fix is described below.
5. Apply the causal mask with `tril` and fill with `-inf`, then softmax, then dropout, then
   `@ v`.

This path is manual: heads run in a Python loop and there is no SDPA or flash attention. The
`Head` class uses `F.scaled_dot_product_attention`, but nothing uses `Head`
any more.

Any `T ≤ block_size` works. Because attention is causal and the bias depends only on
distance, running the first T tokens alone gives the same logits as positions `0…T-1` of a
full window; `tests/test_model.py::test_prefix_logits_match_full_sequence` checks this. (Before
T2, `E` was the whole table and any `T < block_size` crashed, known issue #1.)

### Initialisation (GPT-2 style, `MusicTransformer.__init__` and `_init_weights`)

- All Linear and Embedding weights are drawn from N(0, 0.02), and all biases are set to 0.
- Residual output projections (`*proj.weight` and `mlp.2.weight`) are re-drawn with std
  `0.02 / sqrt(2 · n_layer)`, so the residual stream's variance doesn't grow with depth.

## Parameter count

The formula below is verified against the code: building the 24-layer config gives exactly
the **186,426,284** printed in the 2025-12-29 Colab log.

With C = embedding dim, V = vocab, T = block_size, L = layers, H = heads, Dh = C/H:

```
per head    = 3·C·Dh  (qkv)  + T·Dh  (rel_bias)
per block   = H·per_head + (C² + C) proj + (C·3C + 3C) + (3C·C + C) ffn + 4C (2×LN)
total       = V·C + T·C + L·per_block + 2C (LN_f) + C·V + V
```

| Config | V | C | L | H | T | Params | of which `rel_bias` |
|---|---|---|---|---|---|---|---|
| `hyperparameters.py` defaults | 3000 | 384 | 8 | 4 | 512 | 15.9M | 1.6M |
| 2025-12-03/27 Colab runs | 3500 | 768 | 12 | 12 | 2048 | ~97M | 18.9M |
| 2025-12-29 and 2026-01-31 runs (current) | 3500 | 768 | 24 | 12 | 2048 | **186.4M** | **37.7M (20%)** |

The `rel_bias` tables grow with `block_size × C × L`. They are the reason a 2048 context costs
20% of the model.

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
| Optimizer | AdamW(lr, betas=(0.9, 0.95)), default weight decay 0.01, applied to all params |
| Schedule | `OneCycleLR` over all `epochs`, stepped per optimizer step: 10% warmup from lr/25, then cosine down to lr/25000 |
| Clipping | `clip_grad_norm_(…, --grad_clip)`, default 0.5 |
| Precision | CUDA only: bf16 autocast if supported, else fp16 + GradScaler. TF32 on. MPS/CPU run fp32. |
| Compile | `torch.compile(model)` unless `--no-compile` |
| Device | `--device auto`: cuda → mps → cpu |
| Metrics | One `metrics.jsonl` row per epoch: `train_loss`, `val_loss`, `val_ppl`, `val_bits_per_token`, `lr`, mean `grad_norm` (before clipping), `tokens_per_sec`, `mfu`, `train_time_s`. |
| MFU | `tokens/s × FLOPs/token ÷ peak FLOP/s`. FLOPs/token = `6N + 18·L·C·T`: 6N for the weights (2 forward + 4 backward per parameter) plus three T×C attention matmuls per layer (`q@kᵀ`, `q@Eᵀ`, `attn@v`). Peak is looked up for H100/A100/L4/T4 (dense bf16) or set with `--peak_flops`; otherwise `mfu` is null. |
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

Treat results from before 2026-01-31 as "trained with Q=K=V". They are useful for the stability
lessons, but not as architecture baselines.
