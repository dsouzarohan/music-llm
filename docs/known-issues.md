# Known issues

Tasks in `docs/PLAN.md` refer to issues by number. When an issue is fixed, add
`**Fixed** (<commit>)` to its line, and leave the number in place so references stay valid.

Found by a code read on 2026-10-17. #1 and the parameter count were verified by running the
code under uv (torch 2.14.1, MPS). Nothing here is fixed yet. Numbers are a rough priority
order.

## Correctness

1. **`RelativeHead` crashes when `T < block_size`** (`model.py:88-96`). `rel_bias.weight` is
   always `(block_size, Dh)`, so `.view(B, T+1, T)` fails. Training never triggers this,
   because windows are always full length. `generate()` from a short prompt does trigger it.
   Fix: use `E = self.rel_bias.weight[-T:]`.
2. **`generate()` ignores `top_k`** (`model.py:242`).
3. **Compiled checkpoints carry an `_orig_mod.` key prefix.** `train.py:180` compiles the model
   and `:333` saves `model.state_dict()`. Loading into an uncompiled model (`model_run.ipynb`)
   fails unless the prefix is stripped.
4. **Resume rebuilds `OneCycleLR` from scratch** (`train.py:238`). Warmup restarts, and the
   scheduler overwrites the restored LR. The scheduler state isn't saved.
5. **`final.pt` lacks `vocab_size` and `hyperparams`** (`train.py:373-381`), unlike `best.pt`.
6. **Last partial accumulation group is under-weighted.** Its loss is still divided by the full
   `grad_accum_steps` (`train.py:260`).
7. **Off-by-one in dataset filters.** `GuitarDataset` (`:27-29`) drops the final valid start,
   and both datasets require `len > block_size+1` where `>=` is correct.

## Data

8. **Train/val leakage.** The split is per track, so different guitar parts of one song land in
   both sets (`tokenizer.py:51-79`).
9. **Unseeded split.** `tokenizer.py` deletes and rebuilds `splits/`, `train-midi/`,
   `val-midi/`, `augmented/` and `tokenized/` **at import time**, with an unseeded shuffle.
10. **Val windows are random every epoch** (400 samples), so val loss is noisy.
11. **Artists with commas in their names are dropped** by `csv.reader`
    (`midi_extractor.py:46-68`).
12. **`midi_extractor.py:195`:** `song_file=-midi_file_name` is a unary minus on a str, which
    raises TypeError.
13. **Log block indented inside the `for part` loop** (`midi_extractor.py:235-248`). It logs one
    row per part, and can log `no_guitar_tracks_error` before a later guitar part is found.
14. **Resume checks the wrong status.** It looks for `"no_guitar_tracks"`, but the code writes
    `"no_guitar_tracks_error"`, so no-guitar songs are retried on every resume.
15. **Output filename collisions** when two parts share a `partName`.
16. **Zero-token files** remain in the training set (7 train files and 1 val file in the local
    snapshot).
17. **`dataset_tokenizer.ipynb` cell 3** augments all of `data/midi`, which includes val. If run,
    it leaks augmented val into train.

## Paths / ergonomics

18. **Wrong default paths.** The defaults `../../data` (`train.py:44,48`, `tokenizer.py:15`,
    `midi_extractor.py:14`) assume cwd is `src/utils`. That conflicts with the `src.*` imports,
    which need cwd to be the repo root. Always pass `--data_root data --checkpoint_dir data/checkpoints`.
19. **`scripts/stats.sh` is not runnable.** It has no shebang, uses Jupyter `!python`, contains
    hardcoded `/home/piffs/...` paths, and passes `--vocab_size 3500` while the local tokenizer
    has 3000.
20. **Unused folder paths** in `train.py:78-85`, including the `train_midi` vs `train-midi`
    typo.

## Hygiene / perf

21. **Dead or stale code in `model.py`:**
    - unused hyperparameters import
    - unused `Head` class
    - `FeedForward.dropout` is unused
    - the stale "scalar relative bias" comment
    - hardcoded `mlp_ratio=3`
    - no `n_embd % n_head` assert
22. **Weight decay is applied to LayerNorm weights, biases and embeddings.** There are no
    decay / no-decay param groups.
23. **Heads run in a Python loop** with a manual softmax. There is no fused or flash path.
24. **Device-specific autocast and logging.** Autocast and GradScaler are hardcoded to
    `"cuda"`, and "Compiling model for GPU" is printed on every device.
25. **`model_run.ipynb` cell 7** plots the old scalar `rel_bias` shape, so it's stale.
