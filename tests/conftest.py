import pytest
import torch

from src.model.model import MusicTransformer
from tests.synthetic import VOCAB_SIZE

TINY = dict(vocab_size=VOCAB_SIZE, n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0)

OLD_BLOCK = dict(pos="huang", norm="layernorm", mlp="gelu", qk_norm=False, tie_embeddings=False)
MODERN = dict(norm="rmsnorm", mlp="swiglu", qk_norm=True, tie_embeddings=True)

# The T5 ablation ladder plus the modern parts on Huang: every switch is tested on and off,
# and each pos mode with every other switch both on and off.
BLOCKS = {
    "old": OLD_BLOCK,
    "rope": {**OLD_BLOCK, "pos": "rope"},
    "rope-rms-qk": {**OLD_BLOCK, "pos": "rope", "norm": "rmsnorm", "qk_norm": True},
    "rope-rms-qk-swiglu": {**OLD_BLOCK, "pos": "rope", "norm": "rmsnorm", "qk_norm": True, "mlp": "swiglu"},
    "modern": {"pos": "rope", **MODERN},
    "huang-modern": {"pos": "huang", **MODERN},
}


@pytest.fixture(params=BLOCKS.values(), ids=BLOCKS.keys())
def tiny_model(request):
    """Seeded tiny MusicTransformer on CPU, in eval mode with dropout off, for each block variant."""
    torch.manual_seed(0)
    return MusicTransformer(**TINY, **request.param).eval()


@pytest.fixture
def full_batch():
    """Random (x, y) batch with T == block_size."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, VOCAB_SIZE, (4, TINY["block_size"]), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (4, TINY["block_size"]), generator=g)
    return x, y


def tiny_config(**block):
    """A train.py-style run config for TINY, as save_checkpoint/model_hyperparams expect."""
    b = {**OLD_BLOCK, "mlp_ratio": 3, **block}
    return {"vocab_size": TINY["vocab_size"], "embedding_dim": TINY["n_embd"], "n_head": TINY["n_head"],
            "n_layer": TINY["n_layer"], "block_size": TINY["block_size"], "dropout": TINY["dropout"], **b}
