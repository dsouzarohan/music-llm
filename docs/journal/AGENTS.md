# AGENTS.md: journal entries

How to write an entry in `docs/journal/`. The root `AGENTS.md` says when to add one.

## Tone

Write it like a lab notebook, not a report. The owner's entries (2025-11 to 2026-01) are the model:
read one or two before writing.

- Relaxed and first person ("we tried", "this helped a lot"). Short paragraphs or loose bullets.
- Tell it as a story: what we were trying, what happened, what we think caused it, what's next.
- Explain a concept in a sentence or two of plain language when it comes up, the way
  2025-12-04 explains gradient explosion. Skip the deep mechanics: those belong in
  `docs/learnings/` or the architecture docs, so link there instead.
- No walls of tables, no long lists of decision numbers, file paths or commit hashes. One small
  table is fine when it's the clearest way to compare runs.

## What every entry still covers

Relaxed doesn't mean vague. Whenever the session touched them, write down:

- **Testing:** what was run to check the change (tests, smoke run, a comparison) and whether it
  passed.
- **Model architecture:** what the model looked like for the runs: size, layers/width/heads,
  context, and which block (old or modern pieces).
- **Training numbers:** val loss and ppl (the best epoch, and the last if different), train loss,
  tokens trained on, and the hardware it ran on. A snippet of the raw training log is welcome.
- **Inference numbers:** time per sample, sample length, and grammar error rate when sampling
  was done.
- **Next steps:** a short list at the end.

Keep log output and code snippets as they came out of the terminal.
