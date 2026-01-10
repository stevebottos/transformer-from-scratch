"""
Step 2: Attention Mechanisms

Implement the following:
1. MultiHeadAttention:
    - Scaled Dot-Product Attention (manual implementation).
    - Multi-head splitting and concatenation using `einops`.
    - Linear projections for Q, K, and V.
    - Masking support (optional mask tensor).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        d_emb = k.shape[-1]
        scale = math.sqrt(d_emb)
        scores = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        context = attn @ v
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

        self.sdpa = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q_heads = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k_heads = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v_heads = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        context_heads = self.sdpa(q_heads, k_heads, v_heads, mask)
        context = rearrange(context_heads, "b h s d -> b s (h d)", h=self.n_heads)
        context = self.W_o(context)
        return context


if __name__ == "__main__":
    # 1. Setup dimensions
    emb_dim = 64
    n_heads = 8
    seq_len = 16
    batch_size = 2

    # 2. Create inputs
    q = torch.rand(batch_size, seq_len, emb_dim)
    k = torch.rand(batch_size, seq_len, emb_dim)
    v = torch.rand(batch_size, seq_len, emb_dim)

    # 3. Initialize module
    attn = MultiHeadAttention(emb_dim, n_heads)

    # 4. Test without mask
    out_no_mask = attn(q, k, v)
    print(f"Output shape (no mask): {out_no_mask.shape}")
    assert out_no_mask.shape == (batch_size, seq_len, emb_dim)

    # 5. Test with Causal Mask
    # Create a lower triangular mask (1s on and below diagonal, 0s above)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    out_masked = attn(q, k, v, mask=mask)
    print(f"Output shape (with mask): {out_masked.shape}")
    assert out_masked.shape == (batch_size, seq_len, emb_dim)

    # 6. Verify Causal Mask (The "Tutor" Check)
    # If the mask works, changing the LAST token of the input
    # should NOT change the FIRST token of the output.
    v_modified = v.clone()
    v_modified[:, -1, :] += 1.0  # Modify only the last token

    out_modified = attn(q, k, v_modified, mask=mask)

    # Check if first token output is identical
    diff = (out_masked[:, 0, :] - out_modified[:, 0, :]).abs().max()
    print(
        f"Difference in first token output after modifying last input token: {diff.item():.6f}"
    )
    assert diff < 1e-6, "Causal mask failed! Future tokens influenced the past."
    print("Causal mask verification passed!")
