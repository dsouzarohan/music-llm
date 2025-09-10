import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.hyperparameters import BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM, N_LAYER, N_HEAD, DROPOUT, VOCAB_SIZE

# hyper-parameters
batch_size = BATCH_SIZE
block_size = BLOCK_SIZE
n_embd = EMBEDDING_DIM
vocab_size = VOCAB_SIZE
n_layer = N_LAYER
n_head = N_HEAD
dropout = DROPOUT


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) ** self.head_size ** -0.5  # divide by root of dk as per AIAYN
        # wei here is the attention scores matrix, that ends up being a (T, T) matrix
        # this is why attention computation grows quadratically with context window
        # O(T^2)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.wei = wei
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.heads = nn.ModuleList([
            Head(self.head_size) for _ in range(self.num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # out = self.heads(x) # Just see what happens if we do this, mostly won't work
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)  # Regularization
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.mmha = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = self.ln1(x)
        out = self.mmha(out)
        out = out + x  # skip connections

        out = self.ln2(out)
        out = self.ffwd(out)
        out = out + x  # skip connections
        return out


class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.last = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(torch.arange(T))

        x = (tok_emb + pos_emb)  # add the embeddings to form a new positional aware embedding

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.last(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # stretch all tokens for a batch into a single row
            targets = targets.view(B * T)  # corresponding true classes
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=500):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # important, we only care of the last step
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
