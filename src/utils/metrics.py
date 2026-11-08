"""Throughput and model FLOPs utilisation (MFU)."""

import torch
from torch import nn

# dense bf16/fp16 tensor-core peak, FLOP/s
PEAK_FLOPS = {
    "H100": 989e12,
    "A100": 312e12,
    "L4": 121e12,
    "T4": 65e12,
}


def flops_per_token(model):
    """Training FLOPs per token of an (uncompiled) MusicTransformer at full context.

    6 FLOPs per nn.Linear weight (2 forward, 4 backward; a tied lm_head counts once). Embedding
    lookups, rel_bias, norms and biases are not counted. Attention adds, per layer, 6·T·C for each
    (T x C)-sized matmul per token: q@kT and attn@v, plus q@ET for pos="huang".
    """
    matmul_params = sum({id(m.weight): m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear)}.values())
    attn_matmuls = 3 if model.pos == "huang" else 2
    return 6 * matmul_params + 6 * attn_matmuls * model.n_layer * model.n_embd * model.block_size


def device_peak_flops(device):
    """Peak FLOP/s for a known CUDA GPU, else None."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in PEAK_FLOPS.items() if k in name), None)


def mfu(tokens_per_sec, flops_per_tok, peak_flops):
    """Achieved FLOP/s as a fraction of peak, or None when the peak is unknown."""
    if not peak_flops:
        return None
    return tokens_per_sec * flops_per_tok / peak_flops
