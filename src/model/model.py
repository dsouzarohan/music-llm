import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.hyperparameters import BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM, N_LAYER, N_HEAD, DROPOUT

# hyper-parameters
batch_size = BATCH_SIZE
block_size = BLOCK_SIZE
n_embd = EMBEDDING_DIM
n_layer = N_LAYER
n_head = N_HEAD
dropout = DROPOUT


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
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
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.wei = wei
        wei = self.dropout(wei)
        out = wei @ v
        return out
