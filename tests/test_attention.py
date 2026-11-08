import pytest
import torch

from src.model.model import Attention, apply_rope, rope_tables
from tests.reference_attention import ReferenceAttention


@pytest.mark.parametrize("T", [16, 7, 1])
def test_batched_huang_matches_per_head_reference(T):
    torch.manual_seed(0)
    attn = Attention(n_embd=32, n_head=4, block_size=16, dropout=0.0, pos="huang").eval()
    torch.nn.init.normal_(attn.proj.bias)
    ref = ReferenceAttention(n_head=4, n_embd=32, block_size=16)
    ref.copy_from(attn)
    x = torch.randn(3, T, 32)
    with torch.no_grad():
        assert torch.allclose(attn(x), ref(x), atol=1e-5)


def test_rope_score_depends_only_on_distance():
    torch.manual_seed(0)
    cos, sin = rope_tables(block_size=64, head_size=16)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        return apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])

    assert torch.allclose(score(5, 2), score(60, 57), atol=1e-5)
    assert not torch.allclose(score(5, 2), score(5, 4), atol=1e-3)


def test_rope_preserves_norm():
    cos, sin = rope_tables(block_size=64, head_size=16)
    x = torch.randn(2, 4, 64, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
