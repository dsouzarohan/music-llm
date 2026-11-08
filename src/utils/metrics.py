"""Throughput and model FLOPs utilisation (MFU)."""

import torch

# dense bf16/fp16 tensor-core peak, FLOP/s
PEAK_FLOPS = {
    "H100": 989e12,
    "A100": 312e12,
    "L4": 121e12,
    "T4": 65e12,
}


def flops_per_token(n_params, n_layer, n_embd, block_size):
    """Training FLOPs per token: 6N for the weights plus the attention score matmuls.

    Each layer does three (T x C)-sized matmuls per token in forward (q@kT, q@ET, attn@v),
    2 FLOPs per multiply-add, and backward costs twice forward: 3 * 3 * 2 * T * C.
    """
    return 6 * n_params + 18 * n_layer * n_embd * block_size


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
