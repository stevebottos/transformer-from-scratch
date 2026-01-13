"""
Step 2: Attention Mechanisms

Implement the following:
1. MultiHeadAttention:
    - Scaled Dot-Product Attention (manual implementation).
    - Multi-head splitting and concatenation using `einops`.
    - Linear projections for Q, K, and V (bias=False).
    - Masking support (optional mask tensor).
    - **RoPE (Rotary Positional Embeddings):** Apply rotary embedding to Q and K before attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class ScaledDotProductAttention(nn.Module):
    """Manual implementation of scaled dot-product attention."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, is_causal=False):
        d_emb = k.shape[-1]
        scale = math.sqrt(d_emb)
        scores = (q @ k.transpose(-2, -1)) / scale

        if is_causal:
            seq_len = q.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        context = attn @ v
        return context


class ScaledDotProductAttentionTorch(nn.Module):
    """PyTorch's optimized SDPA with Flash Attention support."""

    def forward(self, q, k, v, mask=None, is_causal=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, use_torch_sdpa=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads

        self.W_qkv = torch.nn.Linear(d_model, d_model * 3, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_torch_sdpa:
            self.sdpa = ScaledDotProductAttentionTorch()
        else:
            self.sdpa = ScaledDotProductAttention()

    def apply_rope(self, x, freqs_cos, freqs_sin):
        """
        Apply Rotary Positional Embeddings using real-valued sin/cos.
        x: (batch, n_heads, seq_len, d_head)
        freqs_cos, freqs_sin: (seq_len, d_head) - Precomputed cos/sin values
        """
        # Reshape freqs for broadcasting: (seq_len, d_head) -> (1, 1, seq_len, d_head)
        cos = freqs_cos.unsqueeze(0).unsqueeze(0)
        sin = freqs_sin.unsqueeze(0).unsqueeze(0)

        # Rotate pairs: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
        x_rotated_odd = x_even * sin[..., 1::2] + x_odd * cos[..., 1::2]

        # Interleave back
        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
        return x_out

    def forward(self, x, freqs_cos=None, freqs_sin=None, is_causal=False):
        qkv = self.W_qkv(x)
        q, k, v = rearrange(
            qkv, "b s (three h d) -> three b h s d", three=3, h=self.n_heads
        )

        # Apply RoPE if provided
        if freqs_cos is not None and freqs_sin is not None:
            q = self.apply_rope(q, freqs_cos, freqs_sin)
            k = self.apply_rope(k, freqs_cos, freqs_sin)

        context_heads = self.sdpa(q, k, v, is_causal=is_causal)
        context = rearrange(context_heads, "b h s d -> b s (h d)", h=self.n_heads)
        context = self.W_o(context)
        return self.dropout(context)


if __name__ == "__main__":
    # 1. Setup dimensions
    emb_dim = 64
    n_heads = 8
    seq_len = 16
    batch_size = 2
    d_head = emb_dim // n_heads

    # 2. Create input
    x = torch.rand(batch_size, seq_len, emb_dim)

    # 3. Initialize module
    attn = MultiHeadAttention(emb_dim, n_heads)

    # 4. Generate RoPE frequencies (real cos/sin)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(seq_len).float()
    freqs_outer = torch.outer(t, freqs)  # (seq_len, d_head/2)
    freqs_outer = freqs_outer.repeat_interleave(2, dim=-1)  # (seq_len, d_head)
    freqs_cos = freqs_outer.cos()
    freqs_sin = freqs_outer.sin()

    # 5. Test with RoPE
    out_rope = attn(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
    print(f"Output shape (with RoPE): {out_rope.shape}")
    assert out_rope.shape == (batch_size, seq_len, emb_dim)

    print("RoPE Attention verification passed!")
