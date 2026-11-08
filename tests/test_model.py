import pytest
import torch

from src.model.model import MusicTransformer, swiglu_hidden
from src.utils.metrics import flops_per_token
from tests.conftest import BLOCKS, OLD_BLOCK, TINY


def expected_params(V, C, L, H, T, pos="huang", norm="layernorm", mlp="gelu", qk_norm=False,
                    tie_embeddings=False, mlp_ratio=3):
    """Parameter-count formula from docs/architecture/model-and-training.md."""
    Dh = C // H
    norm_p = 2 * C if norm == "layernorm" else C
    attn = 3 * C * C + (C * C + C) + (T * C if pos == "huang" else 0) + (2 * Dh if qk_norm else 0)
    if mlp == "gelu":
        h = int(mlp_ratio * C)
        ffn = (C * h + h) + (h * C + C)
    else:
        h = swiglu_hidden(C, mlp_ratio)
        ffn = (C * 2 * h + 2 * h) + (h * C + C)
    per_block = attn + ffn + 2 * norm_p
    embeddings = V * C + (T * C if pos == "huang" else 0)
    lm_head = (0 if tie_embeddings else C * V) + V
    return embeddings + L * per_block + norm_p + lm_head


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_param_count_matches_formula(tiny_model):
    m = tiny_model
    assert n_params(m) == expected_params(
        m.vocab_size, m.n_embd, m.n_layer, m.n_head, m.block_size,
        m.pos, m.norm, m.mlp, m.qk_norm, m.tie_embeddings, m.mlp_ratio,
    )


def test_param_count_matches_colab_186m():
    with torch.device("meta"):
        model = MusicTransformer(vocab_size=3500, n_embd=768, n_head=12, n_layer=24, block_size=2048)
    assert n_params(model) == 186_426_284


def test_swiglu_matches_gelu_param_count_at_186m_shape():
    with torch.device("meta"):
        gelu = MusicTransformer(vocab_size=3500, n_embd=768, n_head=12, n_layer=24, block_size=2048)
        swiglu = MusicTransformer(vocab_size=3500, n_embd=768, n_head=12, n_layer=24, block_size=2048, mlp="swiglu")
    # only the gate/value biases differ: one extra bias vector of size hidden per layer
    assert n_params(swiglu) - n_params(gelu) == 24 * (swiglu_hidden(768, 3) * 2 - 3 * 768)


def test_checkpoint_has_no_buffers(tiny_model):
    """#26: the state dict holds parameters only (no causal masks, no RoPE tables)."""
    assert set(tiny_model.state_dict()) == {n for n, _ in tiny_model.named_parameters(remove_duplicate=False)}


def test_tied_embeddings_share_one_tensor():
    m = MusicTransformer(**TINY, tie_embeddings=True)
    assert m.last.weight is m.token_embeddings.weight


def test_weight_decay_only_on_linear_weights(tiny_model):
    decay, no_decay = tiny_model.param_groups(0.1)
    names = {id(p): n for n, p in tiny_model.named_parameters()}
    decayed = {names[id(p)] for p in decay["params"]}
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(tiny_model.parameters()))
    assert decayed and all(n.endswith("weight") for n in decayed)
    assert not any(k in n for n in decayed for k in ("embeddings", "rel_bias", "ln", "norm"))
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.1, 0.0)


def test_flops_per_token_counts_attention_matmuls_not_lookups():
    C, L, T = TINY["n_embd"], TINY["n_layer"], TINY["block_size"]
    huang = MusicTransformer(**TINY, **OLD_BLOCK)
    rope = MusicTransformer(**TINY, **{**OLD_BLOCK, "pos": "rope"})
    tied = MusicTransformer(**TINY, **{**OLD_BLOCK, "pos": "rope", "tie_embeddings": True})
    assert flops_per_token(huang) - flops_per_token(rope) == 6 * L * C * T  # the q@E^T matmul
    assert flops_per_token(tied) == flops_per_token(rope)


@pytest.mark.parametrize("bad", [dict(pos="alibi"), dict(norm="batchnorm"), dict(mlp="relu")])
def test_unknown_switch_values_raise(bad):
    with pytest.raises(ValueError):
        MusicTransformer(**TINY, **bad)


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
