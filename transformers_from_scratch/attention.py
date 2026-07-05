"""
Mostly lore-accurate Attention Is All You Need (https://arxiv.org/abs/1706.03762)
attention mechanics. Decoder layers (see layers.py) compose these with RMSNorm
and SwiGLUFeedForward into full blocks.
"""

import math

import torch
from torch import nn

__all__ = ["ScaledDotProductAttention", "MultiHeadAttention"]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        D = k.shape[-1]  # batchsize, sequence length, embedding dim
        scale = math.sqrt(D)

        # Using math since D is just a scalar, no need to turn it into a tensor for torch.sqrt
        scores = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            # Assuming that the mask is binary and of the same shape as the input
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # -1 dim because we want to softmax along k-dim
        attn = torch.nn.functional.softmax(scores, dim=-1)

        return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.0):
        super().__init__()
        if not d_model % n_heads == 0:
            raise ValueError("d_model must be divisible by n_heads.")

        if n_kv_heads:
            if not n_heads % n_kv_heads == 0:
                raise ValueError("n_heads must be divisible by n_kv_heads.")

        self.q_dim = d_model  # BEFORE splitting into heads
        self.q_heads = n_heads
        self.heads_dim = d_model // n_heads

        if n_kv_heads:
            # Need to scale down the embedding before splitting, so that when you split
            # with fewer heads than q, you end up with the same embedding shape
            self.kv_dim = self.heads_dim * n_kv_heads
            self.kv_heads = n_kv_heads
        else:
            self.kv_dim = self.q_dim
            self.kv_heads = n_heads

        self.W_qkv = nn.Linear(d_model, self.q_dim + (self.kv_dim * 2), bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.sdpa = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cache=None):

        B, S, _ = x.shape

        qkv: torch.Tensor = self.W_qkv(x)
        q = qkv[:, :, : self.q_dim]
        k = qkv[:, :, self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[:, :, self.q_dim + self.kv_dim :]

        q = q.view(B, S, self.q_heads, self.q_dim // self.q_heads).transpose(1, 2)
        k = k.view(B, S, self.kv_heads, self.kv_dim // self.kv_heads).transpose(1, 2)
        v = v.view(B, S, self.kv_heads, self.kv_dim // self.kv_heads).transpose(1, 2)

        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)  # concat along the sequence
            v = torch.cat((cache[1], v), dim=2)

        if q.size(1) != k.size(1):
            scale_factor = q.size(1) // k.size(1)
            k_attn = k.repeat_interleave(scale_factor, dim=1)
            v_attn = v.repeat_interleave(scale_factor, dim=1)
        else:
            k_attn, v_attn = k, v

        context = self.sdpa(q, k_attn, v_attn, mask)

        # view/reshape merges dims by reading the flat buffer in memory order
        # (rightmost fastest), blind to what the dims mean. Right now memory
        # order is (B, n_heads, S, d_head) - for a fixed seq position, its
        # d_head values are scattered across separate head-blocks, not adjacent.
        # Merging (n_heads, d_head) -> D straight from this layout would grab
        # e.g. head0/seq0 + head0/seq1 as one "token" instead of head0/seq0 +
        # head1/seq0. transpose(1, 2) reorders to (B, S, n_heads, d_head) so
        # each position's heads are contiguous before the merge (reshape
        # instead of .contiguous().view(..) since transpose breaks contiguity).
        context = context.transpose(2, 1).reshape(B, S, self.q_dim)
        return self.dropout(self.W_o(context)), k, v


if __name__ == "__main__":
    B, S, D, n_heads, n_kv_heads = 2, 16, 32, 8, 2

    mha = MultiHeadAttention(D, n_heads, n_kv_heads).eval()
    x = torch.randn(B, S, D)
    mask = torch.tril(torch.ones(S, S))

    out, k, v = mha(x, mask)
    print("out:", out.shape)  # expect (B, S, D)
    print("k:", k.shape)  # expect (B, S, n_kv_heads * d_heads) - the cache saving
    print("v:", v.shape)

    # plain MHA (n_kv_heads=None) should still work unchanged
    mha_plain = MultiHeadAttention(D, n_heads).eval()
    out_plain, k_plain, v_plain = mha_plain(x, mask)
    print("plain k:", k_plain.shape)  # expect (B, S, D)
