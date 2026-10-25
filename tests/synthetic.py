"""Synthetic token corpus in the layout `train.py` reads, so tests never touch `data/`.

Usage: uv run python -m tests.synthetic <data_root>
"""

import json
import random
import sys
from pathlib import Path

VOCAB_SIZE = 64


def write_token_files(folder, n_files, min_len, max_len, vocab_size=VOCAB_SIZE, seed=0):
    """Writes `n_files` miditok-style `{"ids": [...]}` JSON files and returns their paths."""
    rng = random.Random(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_files):
        ids = [rng.randrange(vocab_size) for _ in range(rng.randint(min_len, max_len))]
        path = folder / f"song_{i:03d}.json"
        path.write_text(json.dumps({"ids": ids}))
        paths.append(path)
    return paths


def write_corpus(data_root, vocab_size=VOCAB_SIZE):
    """Writes `tokenized/train-aug/` and `tokenized/val/` under `data_root`."""
    tokenized = Path(data_root) / "tokenized"
    write_token_files(tokenized / "train-aug", 8, 300, 600, vocab_size, seed=0)
    write_token_files(tokenized / "val", 2, 300, 600, vocab_size, seed=1)


if __name__ == "__main__":
    write_corpus(sys.argv[1])
