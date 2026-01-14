"""Multi-head attention with RoPE support."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ScaledDotProductAttention(nn.Module):
    """Manual SDPA implementation."""

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, is_causal=False):
        scale = math.sqrt(k.shape[-1])
        scores = (q @ k.transpose(-2, -1)) / scale

        if is_causal:
            seq_len = q.shape[-2]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn @ v


class ScaledDotProductAttentionTorch(nn.Module):
    """PyTorch SDPA with Flash Attention support."""

    def forward(self, q, k, v, is_causal=False):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        use_torch_sdpa: Use PyTorch's optimized SDPA (Flash Attention).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_torch_sdpa: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads

        self.W_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sdpa = ScaledDotProductAttentionTorch() if use_torch_sdpa else ScaledDotProductAttention()

    def apply_rope(self, x, freqs_cos, freqs_sin):
        """Apply rotary positional embeddings to x."""
        cos = freqs_cos.unsqueeze(0).unsqueeze(0)
        sin = freqs_sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
        x_rotated_odd = x_even * sin[..., 1::2] + x_odd * cos[..., 1::2]

        return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

    def forward(self, x, freqs_cos=None, freqs_sin=None, is_causal=False):
        qkv = self.W_qkv(x)
        q, k, v = rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=self.n_heads)

        if freqs_cos is not None and freqs_sin is not None:
            q = self.apply_rope(q, freqs_cos, freqs_sin)
            k = self.apply_rope(k, freqs_cos, freqs_sin)

        context = self.sdpa(q, k, v, is_causal=is_causal)
        context = rearrange(context, "b h s d -> b s (h d)", h=self.n_heads)
        return self.dropout(self.W_o(context))
