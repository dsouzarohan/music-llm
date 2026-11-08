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

    def huang_mask(self, q):
        """(B, H, T, T) additive mask: the skewed q·E scores scaled by 1/sqrt(Dh), -inf above the diagonal."""
        B, H, T, Dh = q.shape
        E = self.rel_bias[:, -T:, :]  # (H, T, Dh): distances -(T-1)..0
        s_rel = q @ E.transpose(-2, -1)  # (B, H, T, T), columns are distances
        s_rel = F.pad(s_rel, (1, 0)).view(B, H, T + 1, T)[:, :, 1:, :]  # skew: columns become key positions
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        return (s_rel * Dh ** -0.5).masked_fill(~causal, float("-inf"))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)  # each (B, H, T, Dh)
        q, k = self.q_norm(q), self.k_norm(k)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.pos == "rope":
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.huang_mask(q), dropout_p=dropout_p)

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

    def forward(self, x):
        # record_function labels show up in torch.profiler output (train.py --profile_steps)
        with torch.profiler.record_function("attention"):
            x = x + self.attn(self.ln1(x))
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
        self.blocks = nn.Sequential(*[
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embeddings(idx)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings(torch.arange(T, device=idx.device))[None, :, :]

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.last(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=500, temperature=1.0, top_k=None, top_p=None):
        """Appends max_new_tokens sampled tokens to idx (B, T) and returns (B, T + max_new_tokens).

        Any prompt length works; only the last block_size tokens are fed to the model.
        top_k keeps the k most likely tokens; top_p keeps the smallest set of most likely tokens
        whose probabilities sum to at least top_p. Runs in eval mode and restores the previous mode.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)  # important, we only care of the last step
                # we also divide by temperature or 1e-6 to avoid divide by zero
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
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        finally:
            self.train(was_training)
        return idx
