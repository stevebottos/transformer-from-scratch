"""
Decoder layers: RMSNorm and SwiGLUFeedForward, composed with MultiHeadAttention
(see attention.py) into full transformer blocks, dense and MoE.
"""

from dataclasses import dataclass

import torch
from torch import nn

from src.attention import MultiHeadAttention

__all__ = [
    "RMSNorm",
    "SwiGLUFeedForward",
    "LayerOutput",
    "DecoderLayer",
    "NaiveMoEDecoderLayer",
    "MoEDecoderLayer",
]


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


@dataclass
class LayerOutput:
    x: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    aux_loss: torch.Tensor | None = None


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
        return LayerOutput(x, k, v)


class NaiveMoEDecoderLayer(nn.Module):
    """
    "Naive" in two specific ways, both left for the real MoEDecoderLayer below:
    1. No expert capacity cap - each expert processes however many tokens the
       router sends it, so per-expert compute is unbounded and data-dependent
       instead of a fixed, known quantity.
    2. Dispatch is a python loop over experts with boolean masking (see TODO in
       forward()), not a grouped GEMM (sort tokens by expert into contiguous
       chunks, one batched matmul with per-group weights). Correct, but
       n_experts sequential mostly-empty matmuls instead of real vectorized MoE.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_experts,
        ff_expansion=4,
        dropout=0.0,
        max_sequence_length=1024,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff = nn.ModuleList(
            [
                SwiGLUFeedForward(d_model, ff_expansion, dropout)
                for _ in range(n_experts)
            ]
        )
        self.n_experts = n_experts
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_sequence_length, max_sequence_length))
        )
        self.router = nn.Linear(d_model, n_experts)

    def _aux_loss(self, logits, expert_indices):
        """
        Switch-Transformer-style load-balancing loss: n_experts * sum(f_i * P_i).

        P_i: router's average soft confidence per expert (softmax(logits),
        pooled over B, S). Differentiable, but can look near-uniform even when
        actual routing has fully collapsed onto one expert (e.g. every
        token's logits favor the same expert only by a hair).

        f_i: fraction of tokens actually routed to each expert (one-hot of
        expert_indices, pooled over B, S). The real dispatch outcome, but
        comes from argmax so it carries no gradient on its own.

        Multiplying them ties the two together: gradient only flows through
        P_i, and d(loss)/d(P_i) = n_experts * f_i, so an overloaded expert
        (high f_i) gets its probability actively suppressed while an
        underused expert gets almost no pressure. Minimized (loss == 1) when
        both f_i and P_i are uniform (1/n_experts each); collapse onto one
        expert pushes it above 1.

        Worked example, 4 experts / 6 tokens:
            expert_indices: [[0, 0, 3],
                              [0, 2, 0]]        <- 4/6 tokens picked expert 0

            f_i: [0.667, 0.000, 0.167, 0.167]   <- expert 1 never chosen (dead)
            P_i: [0.466, 0.118, 0.209, 0.207]   <- softer, same direction as f_i here,
                                                    but doesn't have to be (that gap is
                                                    exactly what f_i is needed to catch)

            aux_loss = 4 * sum(f_i * P_i) = 1.52   (uniform floor is 1.0)
        """

        # global pooling B, S, n_experts -> n_experts
        # this should look like a uniform distribution, ideally
        probs = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
        n_experts = probs.shape[-1]
        experts_chosen = (
            torch.nn.functional.one_hot(expert_indices, num_classes=n_experts)
            .float()
            .mean(dim=(0, 1))
        )

        # n_experts scales
        return n_experts * (experts_chosen * probs).sum()

    def forward(self, x, cache=None):
        B, S, D = x.shape
        x_out, k, v = self.attn(self.norm1(x), self.mask[:S, :S], cache)
        x = x + x_out
        x_norm = self.norm2(x)

        logits = self.router(x_norm)  # [B, S, n_experts]

        expert_indices = torch.argmax(
            logits, dim=-1
        )  # [B, S] - [<batch>, <seq-idx>] = chosen expert for that token

        aux_loss = None
        if self.training:
            aux_loss = self._aux_loss(logits, expert_indices)

        # TODO: Grouped GEMM dispatch (sort tokens by expert -> contiguous
        # per-expert chunks -> batched matmul with per-group weights) instead
        # of a masked python loop. Needed for real vectorized MoE at scale;
        # masked loop is fine for this repo.
        residual = x
        out = torch.empty(B, S, D, device=x.device, dtype=x.dtype)

        # For every expert, get each point across B,S where we want to use that expert
        for idx in range(self.n_experts):
            mask = expert_indices == idx
            out[mask] = self.ff[idx](x_norm[mask])

        x = out + residual

        return LayerOutput(x, k, v, aux_loss)


class MoEDecoderLayer(NaiveMoEDecoderLayer):
    """
    Subclasses NaiveMoEDecoderLayer to make the delta explicit: same router,
    same experts, same _aux_loss - only dispatch changes (capacity cap +
    grouped GEMM instead of the masked python loop).
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_experts,
        ff_expansion=4,
        dropout=0.0,
        max_sequence_length=1024,
        capacity_factor=1.25,
    ):
        super().__init__(
            d_model, n_heads, n_experts, ff_expansion, dropout, max_sequence_length
        )
        self.capacity_factor = capacity_factor
        self.no_expert_sentinel = -1

    def assign_experts_with_overflow(self, B, S, logits):
        expert_indices = torch.argmax(
            logits, dim=-1
        )  # [B, S] - [<batch>, <seq-idx>] = chosen expert for that token
        fair_capacity = (B * S) / self.n_experts

        counts = torch.bincount(expert_indices.flatten(), minlength=self.n_experts)
        expert_over_share = (counts / fair_capacity) > self.capacity_factor

        if any(expert_over_share):
            expert_probs = torch.max(
                torch.nn.functional.softmax(logits, dim=-1), dim=-1
            )
            raw_experts = expert_indices.clone()
            for idx, is_over in enumerate(expert_over_share):
                if not is_over:
                    continue

                # Now we mask per the index, inverse probs, since we want top-k
                # to return the lowest probs
                masked = 1 - expert_probs.values.clone()
                masked[expert_probs.indices != idx] = float(
                    "-inf"
                )  # so they never appear in the top-k

                masked = masked.view(
                    B * S
                )  # flattening just using view to make it obvious, since I'm using view again later

                trim_num = int(counts[idx] - (fair_capacity * self.capacity_factor))
                trim = torch.topk(masked, k=int(trim_num))
                masked[trim.indices] = self.no_expert_sentinel
                masked = masked.view(B, S)
                expert_indices[masked == self.no_expert_sentinel] = (
                    self.no_expert_sentinel
                )

            return expert_indices, raw_experts

        return expert_indices, expert_indices

    def forward(self, x, cache=None):
        B, S, _ = x.shape

        x_out, k, v = self.attn(self.norm1(x), self.mask[:S, :S], cache)
        x = x + x_out
        x_norm = self.norm2(x)

        logits = self.router(x_norm)  # [B, S, n_experts]
        expert_indices, raw_experts = self.assign_experts_with_overflow(B, S, logits)

        aux_loss = None
        if self.training:
            aux_loss = self._aux_loss(logits, raw_experts)

        residual = x

        # This is what allows passthrough - tokens that weren't touched by
        # an expert are just the original tokens from after attention. Compare with
        # initializing an empty tensor in the naive implementation - that works there because
        # we don't drop overflow tokens from the overloaded expert, every token gets processed
        # here always in that setup
        # TODO: Make this more efficient with a single batched matmul
        out = x_norm.clone()
        for idx in range(self.n_experts):
            mask = expert_indices == idx
            out[mask] = self.ff[idx](x_norm[mask])

        x = out + residual

        return LayerOutput(x, k, v, aux_loss)


if __name__ == "__main__":
    B, S, D, n_heads, n_experts = 2, 16, 32, 4, 4
    layer = MoEDecoderLayer(D, n_heads, n_experts)
    x = torch.randn(B, S, D)
    out = layer(x)
