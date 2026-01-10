"""
Step 1: Embeddings

Implement the following:
1. EmbeddingLayer: 
    - A wrapper around nn.Embedding.
    - Should multiply the output by sqrt(d_model) as per the original paper.
"""

import torch
import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.embedding_layer = nn.Embedding(n_vocab, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens):
        return self.embedding_layer(tokens) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, n_seq, d_model):
        self.embedding_layer = nn.Embedding(n_seq, d_model)

    def forward(self, x): ...


if __name__ == "__main__":
    import tiktoken

    # enc = tiktoken.get_encoding("o200k_base")
    # assert enc.decode(enc.encode("hello world")) == "hello world"

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4o")
    input_embedding = EmbeddingLayer(enc.n_vocab, 512)

    sample = torch.tensor(enc.encode("hello world"))
    print(sample.shape)

    out = input_embedding(sample)
    print(out.shape)
