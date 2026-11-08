import math

import torch
import torch.nn as nn
from torch.nn import functional as F

POS = ("huang", "rope")
NORMS = ("layernorm", "rmsnorm")
MLPS = ("gelu", "swiglu")


def make_norm(kind, dim):
    return nn.LayerNorm(dim) if kind == "layernorm" else nn.RMSNorm(dim, eps=1e-5)


def rope_tables(block_size, head_size, base=10000.0):
    """cos and sin of position × frequency, each (block_size, head_size // 2)."""
    inv_freq = base ** (-torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    angles = torch.outer(torch.arange(block_size, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotates channel pairs (i, i + Dh/2) of x (..., T, Dh) by the angles in cos/sin (T, Dh/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def swiglu_hidden(n_embd, mlp_ratio):
    """SwiGLU hidden size with the same parameter count as a GELU MLP of `mlp_ratio`, rounded up to 64."""
    return 64 * math.ceil(2 * mlp_ratio * n_embd / 3 / 64)


class KVCache:
    """Keys and values of every layer for incremental decoding, preallocated to block_size positions.

    `length` is the number of positions already cached; the model advances it after each forward.
    """

    def __init__(self, n_layer, block_size):
        self.block_size = block_size
        self.length = 0
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def update(self, layer, start, k, v):
        """Writes k, v (B, H, T, Dh) at positions start..start+T-1 of `layer`; returns all cached k, v up to there."""
        if self.k[layer] is None:
            B, H, _, Dh = k.shape
            self.k[layer] = k.new_empty(B, H, self.block_size, Dh)
            self.v[layer] = v.new_empty(B, H, self.block_size, Dh)
        end = start + k.shape[2]
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    """Causal multi-head self-attention, all heads in one F.scaled_dot_product_attention call.

    pos="huang": Music Transformer relative attention. The skewed q·E term is passed to SDPA as an
    additive mask together with the causal mask. pos="rope": rotary embeddings on q and k, plain causal SDPA.
    """

    def __init__(self, n_embd, n_head, block_size, dropout, pos="huang", qk_norm=False):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        if pos not in POS:
            raise ValueError(f"pos must be one of {POS}, got {pos!r}")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.pos = pos
        self.dropout_p = dropout

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)  # rows: [q heads | k heads | v heads]
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.q_norm = nn.RMSNorm(self.head_size, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_size, eps=1e-5) if qk_norm else nn.Identity()

        if pos == "huang":
            # per head, one vector per relative distance; row block_size-1 is distance 0
            self.rel_bias = nn.Parameter(torch.randn(n_head, block_size, self.head_size) * 0.02)
        else:
            cos, sin = rope_tables(block_size, self.head_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def huang_mask(self, q, Tk):
        """Additive mask (B, H, Tq, Tk): the q·E relative scores scaled by 1/sqrt(Dh), -inf above the diagonal.

        Tq == Tk (full window or prefill): skewed (B, H, T, T). Tq == 1 (one decoding step, the query is the
        last key): (B, H, 1, Tk), no skew or causal mask needed.
        """
        B, H, Tq, Dh = q.shape
        E = self.rel_bias[:, -Tk:, :]  # (H, Tk, Dh): distances -(Tk-1)..0
        s_rel = q @ E.transpose(-2, -1)  # (B, H, Tq, Tk), columns are distances
        if Tq == 1:
            return s_rel * Dh ** -0.5
        s_rel = F.pad(s_rel, (1, 0)).view(B, H, Tq + 1, Tq)[:, :, 1:, :]  # skew: columns become key positions
        causal = torch.ones(Tq, Tq, dtype=torch.bool, device=q.device).tril()
        return (s_rel * Dh ** -0.5).masked_fill(~causal, float("-inf"))

    def forward(self, x, cache=None, layer=0):
        """x (B, T, C). With a KVCache, x holds positions cache.length.. and must be either the whole prompt
        (empty cache) or a single token."""
        B, T, C = x.shape
        start = cache.length if cache is not None else 0
        assert cache is None or start == 0 or T == 1, "with a cache, feed the prompt once, then one token per step"
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)  # each (B, H, T, Dh)
        q, k = self.q_norm(q), self.k_norm(k)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.pos == "rope":
            cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, start, k, v)

        if self.pos == "rope":
            # a single query is the newest position, so it may attend to every cached key
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=T > 1)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.huang_mask(q, k.shape[2]), dropout_p=dropout_p)

        out = out.transpose(1, 2).reshape(B, T, C)  # concat heads
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """GELU MLP (hidden mlp_ratio·C) or SwiGLU MLP (hidden swiglu_hidden(C, mlp_ratio), same parameter count)."""

    def __init__(self, n_embd, dropout, mlp="gelu", mlp_ratio=3):
        super().__init__()
        if mlp not in MLPS:
            raise ValueError(f"mlp must be one of {MLPS}, got {mlp!r}")
        self.swiglu = mlp == "swiglu"
        hidden = swiglu_hidden(n_embd, mlp_ratio) if self.swiglu else int(mlp_ratio * n_embd)
        self.up = nn.Linear(n_embd, 2 * hidden if self.swiglu else hidden)  # SwiGLU: [gate | value]
        self.down = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.up(x)
        if self.swiglu:
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        else:
            h = F.gelu(h)
        return self.dropout(self.down(h))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, pos="huang", norm="layernorm", mlp="gelu",
                 qk_norm=False, mlp_ratio=3):
        super().__init__()
        self.attn = Attention(n_embd, n_head, block_size, dropout, pos, qk_norm)
        self.ffwd = FeedForward(n_embd, dropout, mlp, mlp_ratio)
        self.ln1 = make_norm(norm, n_embd)
        self.ln2 = make_norm(norm, n_embd)

    def forward(self, x, cache=None, layer=0):
        # record_function labels show up in torch.profiler output (train.py --profile_steps)
        with torch.profiler.record_function("attention"):
            x = x + self.attn(self.ln1(x), cache, layer)
        with torch.profiler.record_function("mlp"):
            x = x + self.ffwd(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """Pre-norm decoder-only transformer. The block's parts are switches:

    - pos: "huang" (relative attention + absolute position embedding) or "rope" (rotary, no position embedding)
    - norm: "layernorm" or "rmsnorm" (block and final norms)
    - mlp: "gelu" or "swiglu"; mlp_ratio sets the GELU hidden size, SwiGLU matches its parameter count
    - qk_norm: RMSNorm on each head's q and k
    - tie_embeddings: lm_head shares the token embedding weight
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.2,
                 pos="huang", norm="layernorm", mlp="gelu", qk_norm=False, tie_embeddings=False, mlp_ratio=3):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.pos = pos
        self.norm = norm
        self.mlp = mlp
        self.qk_norm = qk_norm
        self.tie_embeddings = tie_embeddings
        self.mlp_ratio = mlp_ratio

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd) if pos == "huang" else None
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size, pos, norm, mlp, qk_norm, mlp_ratio) for _ in range(n_layer)
        ])
        self.ln_f = make_norm(norm, n_embd)
        self.last = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        # GPT-2 scaled init for the residual output projections
        std_scaled = 0.02 / math.sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("attn.proj.weight", "ffwd.down.weight")):
                torch.nn.init.normal_(param, mean=0.0, std=std_scaled)
        if tie_embeddings:
            self.last.weight = self.token_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_groups(self, weight_decay):
        """AdamW groups: weight decay on nn.Linear weights only; none on embeddings (incl. a tied lm_head),
        rel_bias, norms and biases."""
        embedding_ids = {id(m.weight) for m in self.modules() if isinstance(m, nn.Embedding)}
        linear_ids = {id(m.weight) for m in self.modules() if isinstance(m, nn.Linear)} - embedding_ids
        decay = [p for p in self.parameters() if id(p) in linear_ids]
        no_decay = [p for p in self.parameters() if id(p) not in linear_ids]
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def forward(self, idx, targets=None, cache=None):
        """idx (B, T) -> logits (B, T, V) and the mean cross-entropy against targets (or None).

        With a KVCache, idx continues the cached sequence at position cache.length, and the cache advances by T.
        """
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        assert start + T <= self.block_size, f"positions {start}..{start + T - 1} exceed block_size {self.block_size}"
        x = self.token_embeddings(idx)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings(torch.arange(start, start + T, device=idx.device))[None, :, :]

        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        if cache is not None:
            cache.length += T
        x = self.ln_f(x)
        logits = self.last(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=500, temperature=1.0, top_k=None, top_p=None, use_cache=True):
        """Appends max_new_tokens sampled tokens to idx (B, T) and returns (B, T + max_new_tokens).

        Any prompt length works; the model sees at most the last block_size tokens. top_k keeps the k most
        likely tokens; top_p keeps the smallest set of most likely tokens whose probabilities sum to at
        least top_p. Runs in eval mode and restores the previous mode.

        use_cache: process the prompt once, then one token per step against a KVCache. When the window is
        full, the cache is dropped and the last block_size tokens are processed again, so every step sees
        the same window as without the cache.
        """
        was_training = self.training
        self.eval()
        try:
            cache = None
            for _ in range(max_new_tokens):
                if not use_cache:
                    logits, _ = self(idx[:, -self.block_size:])
                elif cache is None or cache.length == self.block_size:
                    cache = KVCache(self.n_layer, self.block_size)
                    logits, _ = self(idx[:, -self.block_size:], cache=cache)
                else:
                    logits, _ = self(idx[:, -1:], cache=cache)
                idx = torch.cat((idx, self.sample_next(logits[:, -1, :], temperature, top_k, top_p)), dim=1)
        finally:
            self.train(was_training)
        return idx

    @staticmethod
    def sample_next(logits, temperature, top_k, top_p):
        """One token id per row (B, 1) from next-token logits (B, V)."""
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            # drop a token once the tokens ranked above it already reach top_p
            drop = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
