"""
Step 4 & 7: Transformer Architecture

Implement the following:
1. DecoderOnlyTransformer (Step 4):
    - Stack of DecoderBlocks.
    - No absolute positional embeddings (RoPE handles this in attention).
    - Final linear layer to vocabulary size (bias=False).
2. EncoderDecoderTransformer (Step 7):
    - Full Transformer including Encoder and Decoder with Cross-Attention.
"""

import math
import torch
import torch.nn as nn
from src.models.layers import DecoderBlock, RMSNorm
from src.models.embeddings import EmbeddingLayer


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_vocab,
        n_layers,
        n_heads,
        expansion_factor=4,
        dropout=0.0,
        max_seq_len=2048,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Precompute RoPE frequencies (cos and sin)
        freqs_cos, freqs_sin = self.precompute_freqs(self.d_head, max_seq_len, rope_theta)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.embedding_input_layer = EmbeddingLayer(n_vocab=n_vocab, d_model=d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, expansion_factor=expansion_factor, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm_final = RMSNorm(d_model=d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)
        self.lm_head.weight = self.embedding_input_layer.embedding_layer.weight

        self._init_weights()

    @staticmethod
    def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
        """
        Precompute cos and sin for Rotary Positional Embeddings.

        Args:
            dim: Dimension of the head (d_head).
            end: Maximum sequence length.
            theta: Scaling factor (default 10000.0).

        Returns:
            freqs_cos, freqs_sin: Tensors of shape (end, dim).
        """
        # Calculate the 'theta' frequencies
        # formula: theta_i = 1.0 / (theta ** (2i / dim))
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Create position indices [0, 1, ..., end-1]
        t = torch.arange(end, dtype=torch.float32)

        # Outer product to get the frequency for every position
        # Shape: (end, dim // 2)
        freqs = torch.outer(t, freqs)

        # Repeat for pairs: (end, dim // 2) -> (end, dim)
        freqs = freqs.repeat_interleave(2, dim=-1)

        return freqs.cos(), freqs.sin()

    def _init_weights(self):
        """
        Initialize weights following common practices:
        - Linear layers: Xavier uniform
        - Embeddings: Normal with std=0.02
        - RMSNorm weights: ones (already default)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, is_causal=True):
        seq_len = x.shape[1]
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        x = self.embedding_input_layer(x)
        x = self.embedding_dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, is_causal=is_causal)

        x = self.norm_final(x)
        return self.lm_head(x)
