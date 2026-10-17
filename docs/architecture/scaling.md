# Current model size, results, and what's next

## Where things stand (as of 2026-01-31)

**Config (Colab A100 80 GB):**

- vocab 3500, context 2048
- 24 layers, 12 heads, d_model 768, dropout 0.1
- batch 128 (micro 8 × 16 accumulation)
- max lr 1e-4, OneCycle schedule, 50 "epochs"
- **186.4M parameters**

**Best result:** val_loss **3.464**, ppl ~32, at epoch 49. Train loss was 3.17. This is the
first run with correct Q/K/V projections (commit 997d968).

**Owner's target:** ppl 5–10 (val_loss ~1.6–2.3). The Music Transformer paper reports about
6.3, on a different dataset and tokenization, so treat that number as a rough north star only.

### Run history (val_loss / ppl)

| Date | Data | Model | Best val | Notes |
|---|---|---|---|---|
| 2025-09-13 | GnR, 33 tracks, vocab 8k | small | 8.9 → 11.3 (val rising) | massive overfit |
| 2025-10-01 | + augmentation | small | 4.68 / 108 | |
| 2025-11-20 | Beatles, ~51M tokens | small, rel-attn | 3.19 / 24 | Q=K=V bug |
| 2025-11-28 | 143 artists, 49M tokens, vocab 3000 | small | 3.03 / 21 | Q=K=V bug |
| 2025-12-27 | same, vocab 3500 | 12L/768/2048 | 4.20 / 67 | Q=K=V bug |
| 2025-12-29 | same | 24L, GELU, GPT-2 init | 3.72 / 41 | Q=K=V bug |
| 2026-01-31 | same | 24L, **Q/K/V fixed** | **3.46 / 32** | current best |

**These numbers are not directly comparable across rows.** A BPE token covers more music at
vocab 3500 than at 3000, so the same music produces fewer tokens, and each token carries more
information and is harder to predict. A ppl of 21 at vocab 3000 is not "better" than 32 at vocab
3500.

To compare runs across vocab sizes, normalise per *base* REMI token. Total nats on the val set
divided by the number of un-BPE'd tokens is a vocab-independent number.

## The size problem: 186M params vs 49M tokens

The model is large relative to its data, and it has had very few optimizer steps. Here are the
numbers.

- **Data per parameter.** Chinchilla's rule of thumb is about 20 tokens per parameter, which
  means ~3.7B tokens for 186M params. The corpus has 49M tokens (about 0.26 per parameter), and
  the augmented copies are near-duplicates, not new music.
- **Repetition.** Muennighoff et al. (2023), "Scaling Data-Constrained LMs", found that
  repeating data is nearly as good as fresh data for about 4 epochs, then returns fall off.
  With ~49M unique tokens × 4 epochs ≈ 200M effective tokens, the compute-optimal model is about
  **10M parameters**, roughly 20× smaller than the current one.
- **Optimizer steps.** 50 epochs × ~16 steps is only **800 optimizer updates**. That is very
  few for a 24-layer model. GPT-2-small trained for hundreds of thousands of steps. The val
  curve was still falling when the LR reached ~0, so the schedule ended before the model
  converged.
- **No overfitting yet.** Train 3.17 vs val 3.46 is a small gap. The model is not memorising,
  which fits "undertrained" better than "too little capacity". Adding layers (12 → 24) bought
  about 0.5 nats, but much of that may simply be more parameters receiving updates in the same
  800 steps.

The cheap experiment that follows from this is to **rebaseline at a smaller size with many
more steps.** For example, compare at equal compute:

| | Small | Current |
|---|---|---|
| Layers | 8 | 24 |
| d_model | 512 | 768 |
| Heads | 8 | 12 |
| Context | 1024 | 2048 |
| Params | ~30M | 186M |
| `epoch_len` | larger, for ~10k+ steps | 2000 (800 steps total) |

If the small model matches or beats 3.46, the bottleneck is data and steps, not depth. That
result would also make every later experiment about 6× cheaper.

## Next steps, roughly in order

`docs/ROADMAP.md` supersedes this list with a phased plan. The list is kept here for the
detail.

1. **Fix the correctness bugs that block evaluation** (`docs/known-issues.md` #1–#5): short-prompt
   generation, `top_k`, the `_orig_mod.` prefix on compiled checkpoints, and scheduler state on
   resume. Without these, sampling from the current best checkpoint is unreliable.
2. **Make val honest:**
   - Split by song, not by track.
   - Seed the split.
   - Use a fixed val set: deterministic windows (`GuitarDataset`) instead of 400 random ones,
     so epoch-to-epoch val is comparable.
3. **Rebaseline small vs large at equal compute** (above). Log to a file or a tracker so runs
   can be compared.
4. **More data.** The journal plan is a FreeMIDI.org scraper targeting the 148-artist list,
   with SQLite logging, retries, rate limiting and resume. That is the only lever that makes a
   ~186M model sensible.
5. **Tokenizer v2:**
   - `num_velocities=32` and a finer `beat_res`, to capture strums and dynamics.
   - Consider pitch bends; guitar bends matter.
   - Insert BOS/EOS at song boundaries.
   - Retraining BPE changes the vocab, so all old checkpoints become incompatible.
6. **Lead-guitar focus** (journal, 2025-10-01): a monophony or "leadness" filter on extracted
   tracks, or a role token such as `<ROLE=LEAD>` for conditioning. The current extraction mixes
   rhythm and lead parts.
7. **Throughput:**
   - Batch all heads into one `(B, H, T, Dh)` op instead of a Python loop over heads.
   - Optionally share `rel_bias` across heads in a layer, which saves up to 37.7M params at the
     current config.
   - Drop the absolute position embedding as an ablation, since relative attention already
     encodes position.
8. **Sampling quality:** implement top-k / top-p, a repetition penalty, and a polyphony guard
   on decode. All are listed in the journal (2025-10-01).
