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
- `rel_bias` is `Embedding(block_size, Dh)`: one learned vector per relative distance
  `0 … -(block_size-1)`.

The forward pass (`model.py:79-120`):

1. `q, k, v` are projected from `x`.
2. `S_rel = q @ Eᵀ` has shape (B, T, block_size). Row *i* holds q_i's affinity to every
   relative distance.
3. **Skew.** Pad one zero column on the left, reshape to (B, T+1, T), and drop the first row.
   This shifts row *i* left by (T-1-i), which turns "query × distance" into
   "query × key-position" without building the (T, T, Dh) tensor that Shaw et al. need.
   That is the memory trick from the Music Transformer paper.
4. Compute `scores = (q @ kᵀ + S_rel) * Dh^-0.5`. Both terms are scaled together; the
   2025-12-04 fix is described below.
5. Apply the causal mask with `tril` and fill with `-inf`, then softmax, then dropout, then
   `@ v`.

This path is manual: heads run in a Python loop and there is no SDPA or flash attention. The
`Head` class (`model.py:18`) uses `F.scaled_dot_product_attention`, but nothing uses `Head`
any more.

> ⚠️ The skew assumes `T == block_size`. Training always satisfies this, because every window
> is exactly `block_size` long. **Any shorter input crashes**, and that includes `generate()`
> from a short prompt. See `docs/known-issues.md` #1.

### Initialisation (GPT-2 style, `model.py:193-215`)

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
| Entry | `python -m src.utils.train` from repo root |
| Data | `{data_root}/tokenized/train-aug/*.json` and `{data_root}/tokenized/val/*.json` |
| Dataset | `RandomGuitarSeqDataset` by default. `__len__` = `epoch_len` (train 2000, val 400). Each item is a random file (weighted by length) at a random offset. `--use_guitar_ds` switches to sliding windows. |
| Batching | `--batch_size` is the effective batch and `--micro_batch_size` is what fits in VRAM. `grad_accum = batch // micro` |
| Optimizer | AdamW(lr, betas=(0.9, 0.95)), default weight decay 0.01, applied to all params |
| Schedule | `OneCycleLR`, stepped per optimizer step: 10% warmup from lr/25, then cosine down to lr/25000 |
| Clipping | `clip_grad_norm_(…, 0.5)` |
| Precision | CUDA only: bf16 autocast if supported, else fp16 + GradScaler. TF32 on. MPS/CPU run fp32. |
| Compile | `torch.compile(model)` whenever available |
| Device | cuda → mps → cpu |
| Eval | full pass over 400 random val windows each epoch. Prints `val_loss`, `ppl`, `bpc`, `Δnats = ln V − loss`, and `x-better = V / ppl` (improvement over uniform guessing) |
| Logging | `print` only (no wandb or tensorboard) |
| Checkpoints | `best.pt` (on val improvement), `epoch_N.pt` (`--save_every`), `final.pt`. The dict holds `model`, `optimizer`, `scaler`, `epoch`, `best_val_loss`, `vocab_size`, and `hyperparams{n_embd,n_head,n_layer,block_size,dropout}`. `final.pt` lacks the last two. |
| Resume | `--resume path.pt` restores model, optimizer, scaler and epoch. The scheduler is **rebuilt** over the remaining epochs, so warmup restarts. |

One epoch here means `epoch_len / micro_batch` micro-steps, **not** one pass over the corpus.
For the current Colab config: 2000 windows / 8 = 250 micro-steps, which at 16× accumulation
is ~16 optimizer steps and ~4.2M tokens. So 50 epochs is 800 optimizer steps and ~210M tokens
seen.

## Inference

`MusicTransformer.generate(idx, max_new_tokens, temperature, top_k)` does plain multinomial
sampling on `logits[:, -1] / temperature`, with the context cropped to `block_size`.

- `top_k` is accepted but not implemented.
- There is no EOS stop.
- The caller must call `.eval()` and wrap the call in `torch.no_grad()`.

`playground/model_run.ipynb` loads `best.pt` and rebuilds the model from
`ckpt["hyperparams"]`. It primes with a 512-token window and decodes with
`REMI(params=data/tokenized/config/tokenizer.json)`, writing the result to `data/output/*.mid`.

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
