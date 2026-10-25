import pytest
import torch

from src.model.model import MusicTransformer
from tests.synthetic import VOCAB_SIZE

TINY = dict(vocab_size=VOCAB_SIZE, n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0)


@pytest.fixture
def tiny_model():
    """Seeded tiny MusicTransformer on CPU, in eval mode with dropout off."""
    torch.manual_seed(0)
    return MusicTransformer(**TINY).eval()


@pytest.fixture
def full_batch():
    """Random (x, y) batch with T == block_size."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB_SIZE, (4, TINY["block_size"]), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (4, TINY["block_size"]), generator=g)
    return x, y
