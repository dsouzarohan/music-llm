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


def test_n_embd_must_divide_by_n_head():
    with pytest.raises(AssertionError, match="divisible"):
        MusicTransformer(**{**TINY, "n_embd": 30})


def test_short_sequence_forward(tiny_model):
    x = torch.randint(0, TINY["vocab_size"], (1, TINY["block_size"] // 2))
    logits, _ = tiny_model(x)
    assert logits.shape == (1, TINY["block_size"] // 2, TINY["vocab_size"])


@pytest.mark.parametrize("T", [1, 5, 15])
def test_prefix_logits_match_full_sequence(tiny_model, full_batch, T):
    x, _ = full_batch
    with torch.no_grad():
        full, _ = tiny_model(x)
        prefix, _ = tiny_model(x[:, :T])
    assert torch.allclose(prefix, full[:, :T], atol=1e-5)


def greedy_tokens(model, out, prompt_len):
    """Argmax of the model's next-token logits at each generated position of `out`."""
    T = model.block_size
    with torch.no_grad():
        return [
            model(out[:, max(0, i - T) : i])[0][0, -1].argmax().item()
            for i in range(prompt_len, out.shape[1])
        ]


@pytest.mark.parametrize("sampling", [dict(top_k=1), dict(top_p=1e-6)])
def test_generate_narrowest_sampling_is_greedy(tiny_model, sampling):
    prompt = torch.randint(0, TINY["vocab_size"], (1, 4))
    out = tiny_model.generate(prompt, max_new_tokens=20, **sampling)
    assert out[0, 4:].tolist() == greedy_tokens(tiny_model, out, 4)


def test_generate_top_k_only_samples_from_top_k(tiny_model):
    torch.manual_seed(0)
    prompt = torch.randint(0, TINY["vocab_size"], (1, TINY["block_size"]))
    with torch.no_grad():
        top3 = tiny_model(prompt)[0][0, -1].topk(3).indices.tolist()
    firsts = {tiny_model.generate(prompt, max_new_tokens=1, top_k=3)[0, -1].item() for _ in range(50)}
    assert firsts <= set(top3) and len(firsts) > 1


def test_generate_from_short_prompt():
    torch.manual_seed(0)
    model = MusicTransformer(**{**TINY, "block_size": 64})
    prompt = torch.randint(0, TINY["vocab_size"], (2, 16))
    out = model.generate(prompt, max_new_tokens=100, temperature=0.9, top_k=40, top_p=0.95)
    assert out.shape == (2, 116)
    assert torch.equal(out[:, :16], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < TINY["vocab_size"]


def test_generate_restores_training_mode(tiny_model):
    tiny_model.train()
    tiny_model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=2)
    assert tiny_model.training
