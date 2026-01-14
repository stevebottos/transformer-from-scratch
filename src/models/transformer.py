"""Decoder-only transformer model."""

import torch
import torch.nn as nn

from src.models.layers import DecoderBlock, RMSNorm
from src.models.embeddings import EmbeddingLayer


class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style decoder-only transformer.

    Args:
        d_model: Model dimension.
        n_vocab: Vocabulary size.
        n_layers: Number of decoder blocks.
        n_heads: Number of attention heads.
        expansion_factor: FFN hidden dim multiplier.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        rope_theta: RoPE base frequency.
    """

    def __init__(
        self,
        d_model: int,
        n_vocab: int,
        n_layers: int,
        n_heads: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads

        freqs_cos, freqs_sin = self._precompute_freqs(self.d_head, max_seq_len, rope_theta)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self.embedding = EmbeddingLayer(n_vocab, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, expansion_factor, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)
        self.lm_head.weight = self.embedding.embedding_layer.weight  # weight tying

        self._init_weights()

    @staticmethod
    def _precompute_freqs(dim: int, seq_len: int, theta: float = 10000.0):
        """Precompute cos/sin for RoPE."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs).repeat_interleave(2, dim=-1)
        return freqs.cos(), freqs.sin()

    def _init_weights(self):
        """Initialize weights: Xavier for linear, normal(0, 0.02) for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, is_causal=True):
        seq_len = x.shape[1]
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        x = self.embedding_dropout(self.embedding(x))
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, is_causal)
        return self.lm_head(self.norm(x))
