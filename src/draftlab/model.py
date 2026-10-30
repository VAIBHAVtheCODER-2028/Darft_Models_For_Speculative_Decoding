"""Fused-QKV GPT with a multi-token cached forward pass and cache rollback.

Stage 1 of domain-draft-lab. Two things this file must get exactly right:
  1. Feeding k new tokens with a cache must give the same logits as a full forward pass.
  2. After rollback(n), the model must behave as if only the first n tokens were ever seen.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 128
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class KVCache:
    """Preallocated key/value store for every layer.

    `length` is the number of valid positions. Rollback only moves `length`;
    stale entries beyond it are never read and get overwritten by the next write.
    """

    def __init__(self, cfg: GPTConfig, batch_size: int = 1, device="cpu", dtype=torch.float32):
        head_dim = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch_size, cfg.n_head, cfg.block_size, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self.max_len = cfg.block_size

    def rollback(self, new_length: int) -> None:
        assert 0 <= new_length <= self.length, f"cannot roll back {self.length} -> {new_length}"
        self.length = new_length


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # Fused QKV: ONE matmul produces Q, K and V for all heads.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, layer_idx: int, cache: KVCache | None = None, past: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): all heads handled in one batched op
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            # write the new chunk after the `past` valid positions, then read everything
            cache.k[layer_idx, :B, :, past:past + T] = k
            cache.v[layer_idx, :B, :, past:past + T] = v
            k = cache.k[layer_idx, :B, :, :past + T]
            v = cache.v[layer_idx, :B, :, :past + T]

        S = past + T  # total keys visible this call
        # Chunk mask: query i sits at absolute position past+i, so it may see keys j <= past+i.
        # tril(diagonal=past) = "all of the past, plus a causal triangle inside the chunk".
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, layer_idx, cache, past):
        x = x + self.attn(self.ln1(x), layer_idx, cache, past)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def new_cache(self, batch_size: int = 1) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, batch_size, device=p.device, dtype=p.dtype)

    def forward(self, idx, cache: KVCache | None = None):
        """idx: (B, T) token ids. Returns logits (B, T, vocab) for EVERY position in idx.

        With a cache, idx is the new chunk only (T may be > 1); positions continue
        from cache.length. The cache length is advanced here, after all layers ran.
        """
        B, T = idx.shape
        past = cache.length if cache is not None else 0
        assert past + T <= self.cfg.block_size, "sequence longer than block_size"

        pos = torch.arange(past, past + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            x = block(x, i, cache, past)
        logits = self.head(self.ln_f(x))

        if cache is not None:
            cache.length = past + T
        return logits