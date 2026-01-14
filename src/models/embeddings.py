"""Token embedding layer."""

import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Token embedding without scaling.

    Note: Original transformer scaled by sqrt(d_model) for sinusoidal positional
    encodings. With RoPE this is unnecessary and omitting simplifies weight tying.
    """

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(n_vocab, d_model)

    def forward(self, tokens):
        return self.embedding_layer(tokens)
