import pytest
import torch

from src.model.model import MusicTransformer
from tests.conftest import TINY


def expected_params(V, C, L, H, T):
    """Parameter-count formula from docs/architecture/model-and-training.md."""
    Dh = C // H
    per_head = 3 * C * Dh + T * Dh
    per_block = H * per_head + (C * C + C) + (C * 3 * C + 3 * C) + (3 * C * C + C) + 4 * C
    return V * C + T * C + L * per_block + 2 * C + C * V + V


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_param_count_matches_formula(tiny_model):
    t = TINY
    assert n_params(tiny_model) == expected_params(
        t["vocab_size"], t["n_embd"], t["n_layer"], t["n_head"], t["block_size"]
    )


def test_param_count_matches_colab_186m():
    with torch.device("meta"):
        model = MusicTransformer(vocab_size=3500, n_embd=768, n_head=12, n_layer=24, block_size=2048)
    assert n_params(model) == 186_426_284


def test_every_parameter_gets_a_gradient(tiny_model, full_batch):
    tiny_model.train()
    _, loss = tiny_model(*full_batch)
    loss.backward()
    missing = [n for n, p in tiny_model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == []


def test_causal(tiny_model, full_batch):
    x, _ = full_batch
    t = 9
    x2 = x.clone()
    x2[:, t] = (x2[:, t] + 1) % TINY["vocab_size"]
    with torch.no_grad():
        a, _ = tiny_model(x)
        b, _ = tiny_model(x2)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)
    assert not torch.allclose(a[:, t], b[:, t])


def test_overfits_one_batch(tiny_model, full_batch):
    tiny_model.train()
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = tiny_model(*full_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="#1 RelativeHead crashes when T < block_size")
def test_short_sequence_forward(tiny_model):
    x = torch.randint(0, TINY["vocab_size"], (1, TINY["block_size"] // 2))
    logits, _ = tiny_model(x)
    assert logits.shape == (1, TINY["block_size"] // 2, TINY["vocab_size"])


@pytest.mark.xfail(strict=True, reason="#2 generate() ignores top_k")
def test_generate_top_k_1_is_greedy(tiny_model):
    T = TINY["block_size"]
    prompt = torch.randint(0, TINY["vocab_size"], (1, T))
    out = tiny_model.generate(prompt, max_new_tokens=5, top_k=1)
    with torch.no_grad():
        for i in range(T, out.shape[1]):
            logits, _ = tiny_model(out[:, i - T : i])
            assert out[0, i] == logits[0, -1].argmax()
