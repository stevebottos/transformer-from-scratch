"""
Mostly lore-accurate Attention Is All You Need (https://arxiv.org/abs/1706.03762)
implementations with some modern additions, notably:
- RMSNorm instead of Layernorm. Dropping bias terms in Linears since RMSNorm
  already mean-centers
- gated SwiGLUFeedForward instead of plain linears
- pre-norm instead of post-norm for better training stability
- our DecoderLayer is actually the encoder layer from the paper, since it's expected
  to be used in a decoder-only setup with a causal mask
"""

import math
import torch
from torch import nn


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
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if not d_model % n_heads == 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        # This is more efficient on GPU than a separate layer for q,k,v
        # because it's one GEMM (GEneral Matrix Multiply instead of three.
        # Each nn.Linear call is one GEMM launch
        self.W_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.sdpa = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cache=None):

        qkv: torch.Tensor = self.W_qkv(x)
        q, k, v = qkv.chunk(3, -1)

        if cache is not None:
            k = torch.cat((cache[0], k), dim=1)
            v = torch.cat((cache[1], v), dim=1)

        bq, sq, dq = q.shape
        bkv, skv, _ = k.shape

        # Reshape to batchsize, n_heads (another batchsize in this case), S, D
        q = q.view(bq, sq, self.n_heads, self.d_heads).transpose(1, 2)
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward network: (Swish(xW1) * xV) @ W2."""

    def __init__(self, d_model: int, ff_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * ff_expansion
        self.W1_plus_gate = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.W2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.W1_plus_gate(x).chunk(2, -1)
        x = self.act(x) * gate
        x = self.W2(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """
    If you look at the paper, this is actually the encoder part of the transformer model
    architecture. However, in decoder-only models, which most are, we use the encoder
    as a decoder (causal mask). The encoder layer counterpart will use a different mask
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ff_expansion=4,
        dropout=0.0,
        max_sequence_length=1024,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, ff_expansion, dropout)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_sequence_length, max_sequence_length))
        )

    def forward(self, x, cache=None):
        B, S, D = x.shape
        x_out, k, v = self.attn(self.norm1(x), self.mask[:S, :S], cache)
        x = x + x_out
        x = self.ff(self.norm2(x)) + x
        return x, k, v
