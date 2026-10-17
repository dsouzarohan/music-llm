# Roadmap

The long-term plan, split into phases. Each phase is one workstream, and the phase in
progress becomes `docs/PLAN.md` as a story with tasks. Background research is in
`docs/architecture/scaling.md` and the "Scaling music-llm" notes (link in `docs/README.md`).

**Direction:** take the project through the full modern LLM stack, from pre-training through
SFT and preference tuning to RL, applied to expressive guitar. NotaGen (pretrain → SFT →
CLaMP-DPO) is the reference recipe. The current bottleneck is data, not compute.

| # | Phase | Status |
|---|---|---|
| 1 | Foundations | **in progress** → `docs/PLAN.md` |
| 2 | Modern block and small rebaseline | planned |
| 3 | Data at scale | planned |
| 4 | Guitar tokenizer | planned |
| 5 | Transfer and SFT | planned |
| 6 | Preference tuning and RL | planned |
| 7 | Performance and kernels | parallel track, after 4 |

## 1. Foundations

Make the training code trustworthy and verifiable before changing what it does.

- A script-driven verification suite (pytest plus a smoke train) that runs locally and in CI.
- Fix the correctness bugs that block sampling and resuming (`docs/known-issues.md`).
- Run config files, per-run metrics logs, and a deterministic val set.
- A first scripted Colab run that measures hardware utilisation (MFU), and a post-fix
  baseline.

**Exit:** `scripts/verify.sh` is green in CI, and a reproducible baseline val loss is recorded.

## 2. Modern block and small rebaseline

Bring the block up to current practice and find out whether depth or data is the bottleneck.

- RoPE in place of absolute pos-emb + skewed `rel_bias`; `scaled_dot_product_attention` on
  `(B,H,T,D)`; RMSNorm with QK-norm; SwiGLU; tied embeddings; weight-decay groups.
- Train a ~30M model with many more optimizer steps. Compare it with the 186M baseline at
  equal compute.

**Exit:** a chosen block, and a measured answer to "small model, more steps vs big model".

**Learn:** the RoPE derivation, FlashAttention, MFU.

## 3. Data at scale

Grow the corpus 20–40× and measure what each source contributes.

- Extraction on symusic using GM program numbers, replacing music21 name matching. Replace
  `scripts/stats.sh` with a scripted data report.
- Lakh MIDI full (~177k files) and GigaMIDI guitar tracks; dedupe by note-content hash.
- An **all-instrument corpus** for pretraining. On Lakh Clean, non-drum instruments have 2.8×
  as many notes as guitar alone.
- Song-level, seeded splits.
- Request DadaGP access. It is available on request from the authors, for research use.
- **Transfer experiment:** guitar-only vs all-instruments → guitar, at equal compute.
- Fit a small scaling law: 3–4 model sizes × 2–3 data sizes.
- Only consider a FreeMIDI scraper after these sources are exhausted, and check its ToS and
  robots.txt first.

**Exit:** a deduplicated, versioned corpus with a data report, and scaling and transfer
curves that decide the next model size.

**Learn:** data-constrained scaling, and scaling laws for transfer.

## 4. Guitar tokenizer

Represent what makes guitar sound like guitar.

- Bends compressed into one event per bend (e.g. `Bend_full`). Today, bends are dropped at
  parse time, even though 32.5% of Lakh guitar tracks have real ones.
- Tab-native tokens (string, fret, slide, palm mute) on DadaGP / PyGuitarPro.
- Ablation: RoPE per musical attribute (onset, pitch class, octave, duration, string),
  following Moonbeam.
- Prototype in Python. Port the hot path to Rust (PyO3 + maturin) once the format is stable.

**Exit:** a tokenizer whose encode → decode round trip keeps bends and techniques, and a model
trained on it.

## 5. Transfer and SFT

Turn the base model into a guitar model you can steer.

- Pretrain on all instruments, then continue training on guitar, mixing in some general data
  to avoid forgetting.
- SFT on a curated lead-guitar set with control tokens (genre, artist, role). Generation is
  primed with 1–2 bars.
- Evals: symbolic statistics (pitch-class entropy, scale consistency, polyphony, empty bars,
  groove, playability) and Fréchet Music Distance.

**Exit:** a controllable guitar model, and an eval suite that runs on every checkpoint.

## 6. Preference tuning and RL

Optimise for music that is good, not just likely.

- An A/B listening page for the owner's labels.
- DPO on chosen/rejected pairs.
- GRPO with verifiable rewards (in-key ratio, fretboard playability, rhythmic consistency)
  plus a KL penalty. Watch for reward hacking.

**Exit:** blind A/B listening prefers the tuned model over the SFT model.

**Learn:** the DPO derivation, GRPO advantages, reward design.

## 7. Performance and kernels

A parallel track, once the architecture has settled after phase 4.

- Profile first.
- PyTorch FlexAttention for attention variants.
- Hand-written Triton kernels for model-specific fused ops.
- CUDA in C++ or Rust (NVIDIA CUDA Rust: cuda-oxide / cutile-rs, Linux only) for learning,
  e.g. a tiled relative-attention kernel.

**Exit:** a measured speedup on the phase-4 model.
