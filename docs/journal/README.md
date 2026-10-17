# Journal

The project's working log: what was tried, what happened, and what came next. There is one
file per entry, named `YYYY-MM-DD.md`.

Entries up to 2026-01-31 were migrated from the Notion page
[music-transformer](https://app.notion.com/p/music-transformer-1cccd7cfa65c801e84d7cfeff67fddcb)
on 2026-10-17. Their text is kept as written. Training logs inside an entry are unedited
terminal output.

New entries go here as new dated files. They record what was done and learned in that
session, while `docs/PLAN.md` tracks the work itself.

**Original project statement (Notion):**

> Project to implement a transformer style neural network that can generate guitar solos
> based on training. As of 2025-09-08, what I plan to make is first the simple transformer
> from scratch using PyTorch.

The Notion page also links its research threads in
[ChatGPT](https://chatgpt.com/g/g-p-67fba7907c408191876a192f8df9495e-prj-music-transformer/project)
and [Gemini](https://gemini.google.com/app/5b612ef64d7b60b9).

## Learnings

- [MIDI tokenization](learnings/midi-tokenization.md): MIDI messages, channels, and how
  time division works.

## Entries

Notes on reading the entries:

- Every relative-attention run before 2026-01-31 used Q = K = V (fixed in commit 997d968).
- Losses at vocab 3000 and vocab 3500 aren't directly comparable. See
  `../architecture/scaling.md`.

| Date | Summary |
|---|---|
| [2025-04-05](2025-04-05.md) | Reading the MidiTok paper and docs; plan to tokenize a single GnR track |
| [2025-04-28](2025-04-28.md) | First MIDI extractor (GnR electric guitar parts) and tokenizer script |
| [2025-09-08](2025-09-08.md) | Restart after nanoGPT; BPE tokenizer trained to vocab 8000 on 33 tracks |
| [2025-09-09](2025-09-09.md) | `utils` package, `GuitarDataset`, first attention head; 67,887 tokens |
| [2025-09-10](2025-09-10.md) | Transformer written from scratch, following "Attention Is All You Need" |
| [2025-09-13](2025-09-13.md) | NaN attention traced to the `**` vs `*` scaling bug; massive overfitting |
| [2025-09-15](2025-09-15.md) | Random-window dataset; reading list on span sampling (GPT-3, Pile, Megatron, PaLM) |
| [2025-09-16](2025-09-16.md) | Perplexity and eval metrics; data augmentation becomes the priority |
| [2025-10-01](2025-10-01.md) | Augmentation helps; first MIDI generated from the model; lead-vs-rhythm ideas |
| [2025-10-18](2025-10-18.md) | Moved to the Ryzen 7 9700X + RTX 5070 Ti box: 1m40s vs 30–45 min on M3 Pro MPS |
| [2025-10-21](2025-10-21.md) | SSH workflow; OOM at 16 layers; dataset scope and relative attention next |
| [2025-10-26](2025-10-26.md) | One MIDI per guitar track; SDPA for FlashAttention; bf16 |
| [2025-11-15](2025-11-15.md) | First relative attention; switched to The Beatles (~51M tokens) |
| [2025-11-18](2025-11-18.md) | Weight checkpointing; notebook loads saved weights |
| [2025-11-20](2025-11-20.md) | Huang/Shaw relative attention done (val 3.19, ppl 24); plan to scale to an A100 |
| [2025-11-22](2025-11-22.md) | Extractor scaled to the 143-artist list |
| [2025-11-23](2025-11-23.md) | MIDI extractor 2.0: fault tolerance, batching, resume, logging |
| [2025-11-26](2025-11-26.md) | Extracted artists 1–40: 5,794 guitar tracks |
| [2025-11-28](2025-11-28.md) | Full extraction: 8,410 MIDIs, 49M train tokens; val 3.03 (ppl 21) |
| [2025-11-30](2025-11-30.md) | A100 OOM: the 2048² attention matrix; gradient accumulation planned |
| [2025-12-01](2025-12-01.md) | Hardcoded batch-size bug fixed; all hparams exposed as CLI args |
| [2025-12-03](2025-12-03.md) | Gradient accumulation works on A100 (61 GB); crash from an f-string format bug |
| [2025-12-04](2025-12-04.md) | Gradient explosions; OneCycleLR, batch 128, lr 1e-4; relative-bias scaling diagnosis |
| [2025-12-08](2025-12-08.md) | Scaled bias with content scores; clip 0.5; resume from `best.pt` |
| [2025-12-27](2025-12-27.md) | Full 50-epoch Colab run (12L): val 4.20 (ppl 67) |
| [2025-12-29](2025-12-29.md) | GELU + GPT-2 init + 24 layers (186M): val 3.72 (ppl 41); data diagnosis |
| [2026-01-31](2026-01-31.md) | Found and fixed Q = K = V; val 3.46 (ppl 32), current best; scraper plan |
