"""
Step 3: Layers & Normalization

Implement the following:
1. RMSNorm:
    - Implement from scratch: x / RMS(x) * weight.
2. SwiGLUFeedForward:
    - Implement SwiGLU: (Swish(xW) * xV)W2
    - Three linear layers (bias=False).
    - Typically hidden_dim = 4 * d_model (or 8/3 * d_model for SwiGLU to match param count).
3. DecoderBlock:
    - Pre-Norm architecture.
    - Sublayer 1: RMSNorm -> RoPE Attention -> Residual Add.
    - Sublayer 2: RMSNorm -> SwiGLU FFN -> Residual Add.
"""

import torch

from src.models.attention import MultiHeadAttention


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rsqrt * self.weight


class SwiGLUFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, expansion_factor: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        self.W1 = torch.nn.Linear(d_model, hidden_dim, bias=False)
        self.gate = torch.nn.Linear(d_model, hidden_dim, bias=False)
        self.W2 = torch.nn.Linear(hidden_dim, d_model, bias=False)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x_pre = self.act(self.W1(x))
        x_v = self.gate(x)
        out = self.W2(x_pre * x_v)
        return self.dropout(out)


class DecoderBlock(torch.nn.Module):
    """
    For decoder-only use.
    """

    def __init__(self, d_model, n_heads, expansion_factor=4, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff = SwiGLUFeedForward(d_model, expansion_factor, dropout=dropout)

    def forward(self, x, freqs_cos=None, freqs_sin=None, is_causal=True):
        # Modern approach applies normalization before entering attn/ff
        y = self.attn(self.norm1(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin, is_causal=is_causal)
        x = x + y
        y = self.ff(self.norm2(x))
        return x + y
