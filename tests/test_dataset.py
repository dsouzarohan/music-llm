import json
import random

import pytest

from src.utils.data.random_guitar_seq_dataset import RandomGuitarSeqDataset
from tests.synthetic import VOCAB_SIZE, write_token_files

BLOCK = 16


@pytest.fixture
def token_files(tmp_path):
    return write_token_files(tmp_path, n_files=4, min_len=40, max_len=80)


def test_windows_are_shifted_slices_of_a_file(token_files):
    random.seed(0)
    ds = RandomGuitarSeqDataset(token_files, block_size=BLOCK, epoch_len=50)
    sources = [json.loads(p.read_text())["ids"] for p in token_files]
    assert len(ds) == 50
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == y.shape == (BLOCK,)
        assert 0 <= int(x.min()) and int(x.max()) < VOCAB_SIZE
        assert x[1:].tolist() == y[:-1].tolist()
        window = x.tolist() + y[-1:].tolist()
        assert any(
            ids[s : s + BLOCK + 1] == window for ids in sources for s in range(len(ids) - BLOCK)
        )


def test_short_files_are_dropped(tmp_path, token_files):
    short = write_token_files(tmp_path / "short", n_files=2, min_len=5, max_len=10, seed=1)
    ds = RandomGuitarSeqDataset(token_files + short, block_size=BLOCK)
    assert ds.num_files == len(token_files)
