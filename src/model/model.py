import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.hyperparameters import BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM, N_LAYER, N_HEAD, DROPOUT, VOCAB_SIZE


# hyper-parameters
# batch_size = BATCH_SIZE
# block_size = BLOCK_SIZE
# n_embd = EMBEDDING_DIM
# vocab_size = VOCAB_SIZE
# n_layer = N_LAYER
# n_head = N_HEAD
# dropout = DROPOUT


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout_p = dropout
        self.block_size = block_size

    def forward(self, x):
        # x: [B, T, C], project to head_dim
        B, T, C = x.shape
        q = self.query(x)# [B, T, Dh]
        k = self.key(x)# [B, T, Dh]
        v = self.value(x)# [B, T, Dh]

        # Writing out a hand rolled QKV attention to PyTorch's inbuilt Scaled Dot-Product Attention (SDPA) unlocks FlashAttention
        # Manually implementing the softmax(QK^T)V is a great way to learn attention, but can become a memory bottleneck

        # wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # divide by root of dk as per AIAYN
        # # wei here is the attention scores matrix, that ends up being a (T, T) matrix
        # # this is why attention computation grows quadratically with context window
        # # O(T^2)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        # out = wei @ v

        # we will use torch's SDPA here
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True # setting this to true ensures transformers don't peek into the future, only depend on past tokens
        )
        return out # returns a single attention head output [B, T, Dh]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, dropout, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads # try to keep head_size/head_dim = 64/96/128

        self.heads = nn.ModuleList([
            Head(self.head_size, n_embd, block_size, dropout) for _ in range(self.num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)  # Regularization
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.mlp_ratio = 3 # known as the MLP ratio, how much we expand the dimensions in the MLP before bring it back down
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * self.mlp_ratio),
            nn.ReLU(),
            nn.Linear(n_embd * self.mlp_ratio, n_embd),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp(x)
        # print("MLP forward", x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.mmha = MultiHeadAttention(n_head, n_embd, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):# recommended
        x = x + self.mmha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

        self.token_embeddings = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embeddings = nn.Embedding(self.block_size, self.n_embd)

        self.blocks = nn.Sequential(
            *[Block(self.n_embd, self.n_head, self.dropout, self.block_size) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.last = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embeddings(idx)
        pos = torch.arange(T, device=idx.device, dtype=torch.long) # ensure positional embeddings are also on the same device
        pos_emb = self.position_embeddings(pos)[None, :, :]

        x = (tok_emb + pos_emb)  # add the embeddings to form a new positional aware embedding

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.last(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # perform reshape for cross entropy
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=500, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # important, we only care of the last step
            # we also divide by temperature or 1e-6 to avoid divide by zero
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
