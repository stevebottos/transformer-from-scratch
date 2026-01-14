"""Transformer building blocks: normalization, feedforward, decoder block."""

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention


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

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        self.W1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.act(self.W1(x)) * self.gate(x)))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: Norm -> Attention -> Add -> Norm -> FFN -> Add."""

    def __init__(self, d_model: int, n_heads: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff = SwiGLUFeedForward(d_model, expansion_factor, dropout=dropout)

    def forward(self, x, freqs_cos=None, freqs_sin=None, is_causal=True):
        x = x + self.attn(self.norm1(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin, is_causal=is_causal)
        x = x + self.ff(self.norm2(x))
        return x
