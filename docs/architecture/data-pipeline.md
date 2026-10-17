# Data extraction pipeline

```
Lakh MIDI Clean (data/music/<Artist>/*.mid)
   │  midi_extractor.py   (music21: partitionByInstrument, keep parts named *guitar*/*gtr*)
   ▼
data/midi/<Artist>_<Song>_<Part>.mid       one file per guitar part
   │  tokenizer.py  ①  random split (unseeded, TRAINING_SPLIT=0.6) → train-midi/, val-midi/, splits/*.txt
   │                ②  miditok augment_dataset(train-midi) → augmented/
   │                ③  REMI + BPE trained on augmented/*.mid → tokenized/config/tokenizer.json
   │                ④  encode → tokenized/train-aug/*.json, tokenized/val/*.json   ({"ids": [...]})
   ▼
train.py → RandomGuitarSeqDataset (random block_size windows)
```

## 1. Source and artist selection

The raw data is the **Lakh MIDI Dataset "Clean"** repack: 2,197 artist folders and 17k
`.mid` files, about 811 MB, in `data/music/`. You have to download it manually; no script
does this.

Two lists decide which artists are used:

- `data/lists/artist_names.txt` lists every folder name.
- `data/lists/training_artist_names.csv` lists 147 hand-picked rock artists, chosen from
  "greatest rock artists" lists (journal, 2025-11-22).

The extractor keeps the intersection of the two lists. Artist names that contain commas
(Earth, Wind & Fire; CSNY; ELP) get split by `csv.reader` and silently dropped.

## 2. Guitar extraction (`src/utils/midi_extractor.py`)

The extractor works in four steps:

1. Parse each MIDI with music21.
2. Call `instrument.partitionByInstrument`.
3. Keep the parts whose `instrumentName` contains "guitar" or "gtr".
4. Write each kept part to its own `.mid`.

Matching is by name only. The GM program number (24–31) is not checked, and music21 merges
parts that share an instrument.

The extractor also handles batching and resuming:

- `--artist-offset N --artist-limit M` or `--artist-name X` choose which artists to process.
- `--resume` and `--clean-output` control what happens to existing output.
- Progress is logged per song to `data/logs/midi_extraction_log.csv`.
- `scripts/stats.sh` holds grep and awk one-liners for summarising that log.

`midi_extractor.py` must be run from `src/utils`, because `DATA_PATH = Path("../../data")`.

**Full run (2025-11-28, Linux box):** 143 artists produced **8,410** guitar MIDIs. The Beatles
alone account for 16k log rows, so the corpus is skewed toward them.

## 3. Tokenization (`src/utils/tokenizer.py`)

The tokenizer is miditok **REMI**, built with
`TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_rests=True)`.
The saved config holds these settings:

- pitch range 21–108
- `beat_res {0-4: 8, 4-12: 4}`
- one token stream for programs
- no tempos, time signatures, pedals or pitch bends
- `encode_ids_split="bar"`, so BPE merges never cross a bar line

**BPE** is trained on the augmented train MIDIs. The base REMI vocab has 441 tokens, and BPE
grows it to `--vocab_size`: 3000 for the local snapshot, 3500 for the Colab corpus. A BPE token
can stand for a whole recurring figure, such as a pitch + velocity + duration group or a common
lick. That's why the same music needs fewer tokens as the vocab grows.

**Pitch bends are dropped** (`use_pitch_bends=False`). A scan of Lakh Clean on 2026-10-17
(symusic, 17,150 files) found:

- 32.5% of guitar tracks have real bends (|value| > 1/32 of range), about 290 bend
  messages each.
- Only 4.7% of piano tracks do. Most bend messages on non-bending instruments are
  zero-value resets.

In Lakh Clean, notes from all non-drum instruments outnumber guitar notes 2.8×; including
drums, 4.2×.

The special tokens are `PAD`, `BOS`, `EOS` and `MASK` (ids 0–3). **None of them is inserted**,
so the token files carry no song-boundary signal.

## 4. Augmentation

The pipeline calls `miditok.data_augmentation.augment_dataset` on `train-midi/` only. It uses
these offsets:

- pitch ±12 semitones
- velocity −4/+5
- duration −0.5/+1 beat

With `all_offset_combinations=False`, each copy varies one dimension at a time. Pitch shifts
that leave the guitar's range are dropped. Copies are named with a suffix:
`#p12`, `#v-4`, `#d10080` (the duration suffix is in ticks, at 10080 ticks per beat).

## 5. Splits

The split is **per track, not per song**. Songs often have 2–5 guitar parts, for example
`November_Rain.1…4` and `November_Rain_CLEAN_GTR`, so different parts of the same song land on
both sides of the split. Val loss is therefore optimistic. This is listed in `docs/known-issues.md`.

`tokenizer.py` re-shuffles with no seed and deletes the old split on every run.

## 6. Datasets (`src/utils/data/`)

**`RandomGuitarSeqDataset`** (the default) works like this:

- Files with `len ≤ block_size+1` are dropped. At a 2048 context that excludes many short
  tracks.
- Each `__getitem__` ignores its index. It picks a file with probability proportional to the
  number of valid start positions, picks a uniform random start, and returns `(x, y)` shifted
  by one token.

These are the "random contiguous span" sampling ideas from GPT-3 and Megatron.

**`GuitarDataset`** produces deterministic sliding windows with stride `block_size // 2`.

## Corpus sizes

| Snapshot | Where | Artists | Source tracks | Train files (aug) | Val files | Train tokens | Val tokens | Vocab |
|---|---|---|---|---|---|---|---|---|
| Local (stale) | `data/` on this Mac | 2 (GnR, Green Day) | 123 | 521 | 25 | ~1.67M | ~95K | 3000 |
| Current | Google Drive `MyDrive/Projects/music-llm/data` | 143 | 8,410 | 17,153 | 3,365 | ~49.2M | ~6.0M | 3500 |

The local copy also has stale data. These items are leftovers from older pipeline versions:

- 33 top-level `data/tokenized/*.json` files
- a `data/midi/` folder that is out of sync with the splits
- 8 files that tokenize to 0 tokens

## Reproducing from scratch

1. Put Lakh MIDI Clean in `data/music/`, and the lists in `data/lists/`.
2. Extract the guitar tracks:
   `cd src/utils && uv run python midi_extractor.py --artist-offset 0 --artist-limit 40 --resume`
   Repeat with later offsets until all artists are done.
3. Split, augment, train BPE and tokenize:
   `uv run python src/utils/tokenizer.py --data_root data/ --vocab_size 3500`
   Run this from the repo root, with `PYTHONPATH=.` if the `src` import fails.
4. Train: see `docs/architecture/model-and-training.md` and `docs/colab.md`.
