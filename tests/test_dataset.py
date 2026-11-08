import json
import random

import pytest
import torch

from src.utils.data.guitar_dataset import GuitarDataset
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


def test_file_of_exactly_block_plus_one_tokens_is_kept(tmp_path):
    exact = write_token_files(tmp_path, n_files=1, min_len=BLOCK + 1, max_len=BLOCK + 1)
    assert RandomGuitarSeqDataset(exact, block_size=BLOCK).num_files == 1
    assert len(GuitarDataset(block_size=BLOCK, stride=1, file_list=exact)) == 1


def test_guitar_dataset_includes_last_valid_start(tmp_path):
    files = write_token_files(tmp_path, n_files=1, min_len=BLOCK + 5, max_len=BLOCK + 5)
    ids = json.loads(files[0].read_text())["ids"]
    ds = GuitarDataset(block_size=BLOCK, stride=1, file_list=files)
    assert len(ds) == 5
    assert ds[4][1].tolist() == ids[-BLOCK:]


def test_guitar_dataset_max_windows_is_deterministic(token_files):
    a = GuitarDataset(block_size=BLOCK, stride=BLOCK, file_list=token_files, max_windows=5)
    b = GuitarDataset(block_size=BLOCK, stride=BLOCK, file_list=token_files, max_windows=5)
    assert len(a) == 5
    assert all(torch.equal(a[i][0], b[i][0]) for i in range(5))
