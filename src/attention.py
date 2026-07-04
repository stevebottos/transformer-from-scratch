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

        # TODO(GQA): n_kv_heads=None should fall back to n_heads (plain MHA,
        # today's behavior). Otherwise n_heads must be divisible by
        # n_kv_heads - n_heads // n_kv_heads query heads share each K/V head.
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        # This is more efficient on GPU than a separate layer for q,k,v
        # because it's one GEMM (GEneral Matrix Multiply instead of three.
        # Each nn.Linear call is one GEMM launch
        # TODO(GQA): K/V only need n_kv_heads * d_heads each, not d_model each -
        # W_qkv's output width shrinks accordingly.
        self.W_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.sdpa = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cache=None):

        # TODO(GQA): split q vs k/v unevenly now (q gets d_model, k/v get
        # n_kv_heads * d_heads each) instead of one even chunk(3, -1).
        qkv: torch.Tensor = self.W_qkv(x)
        q, k, v = qkv.chunk(3, -1)

        # cache should hold the small n_kv_heads K/V, not the expanded ones -
        # that's where the memory saving comes from. Keep the cache concat
        # above wherever the expansion step ends up.
        # TODO(pre-GQA cleanup): cache is tracked pre-reshape (flat (B,S,D))
        # right now, so every call reshapes the whole concatenated history,
        # not just the new token. Standard practice (HF past_key_values,
        # llama.cpp) caches post-reshape, per-head (B,n_heads,S,d_head), so
        # only the new step's K/V needs reshaping before concat. Fix this
        # ordering before building GQA on top of it.
        if cache is not None:
            k = torch.cat((cache[0], k), dim=1)
            v = torch.cat((cache[1], v), dim=1)

        bq, sq, dq = q.shape
        bkv, skv, _ = k.shape

        # Reshape to batchsize, n_heads (another batchsize in this case), S, D
        q = q.view(bq, sq, self.n_heads, self.d_heads).transpose(1, 2)
        # TODO(GQA): k/v reshape to n_kv_heads here, then expand
        # (repeat_interleave) each K/V head across its n_rep query heads
        # before they hit sdpa.
        _k = k.view(bkv, skv, self.n_heads, self.d_heads).transpose(1, 2)
        _v = v.view(bkv, skv, self.n_heads, self.d_heads).transpose(1, 2)

        context = self.sdpa(q, _k, _v, mask)
        # view/reshape merges dims by reading the flat buffer in memory order
        # (rightmost fastest), blind to what the dims mean. Right now memory
        # order is (B, n_heads, S, d_head) - for a fixed seq position, its
        # d_head values are scattered across separate head-blocks, not adjacent.
        # Merging (n_heads, d_head) -> D straight from this layout would grab
        # e.g. head0/seq0 + head0/seq1 as one "token" instead of head0/seq0 +
        # head1/seq0. transpose(1, 2) reorders to (B, S, n_heads, d_head) so
        # each position's heads are contiguous before the merge (reshape
        # instead of .contiguous().view(..) since transpose breaks contiguity).
        context = context.transpose(2, 1).reshape(bq, sq, dq)
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
