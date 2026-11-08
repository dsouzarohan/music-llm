"""Frozen copy of the pre-Phase-2 per-head Huang attention (model.py RelativeHead + MultiHeadAttention).

Test oracle only: tests/test_attention.py checks the batched SDPA rewrite against it.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class RelativeHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.rel_bias = nn.Embedding(block_size, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        E = self.rel_bias.weight[-T:]
        s_rel = q @ E.transpose(-2, -1)
        s_rel = F.pad(s_rel, (1, 0)).view(B, T + 1, T)[:, 1:, :]
        scores = (q @ k.transpose(-2, -1) + s_rel) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class ReferenceAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([RelativeHead(n_embd // n_head, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

    @torch.no_grad()
    def copy_from(self, attn):
        """Copies weights from a batched `src.model.model.Attention` (pos="huang") into the per-head layout."""
        C = attn.qkv.in_features
        Dh = attn.head_size
        W = attn.qkv.weight  # (3C, C): rows [q heads | k heads | v heads]
        for h, head in enumerate(self.heads):
            rows = slice(h * Dh, (h + 1) * Dh)
            head.query.weight.copy_(W[rows])
            head.key.weight.copy_(W[C:][rows])
            head.value.weight.copy_(W[2 * C:][rows])
            head.rel_bias.weight.copy_(attn.rel_bias[h])
        self.proj.load_state_dict(attn.proj.state_dict())
