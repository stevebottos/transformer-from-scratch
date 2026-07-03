"""Decoder-only transformer model."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from src.layers.attention import DecoderLayer, RMSNorm


class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style decoder-only transformer.
    """

    def __init__(
        self,
        d_model: int,
        n_vocab: int,
        n_layers: int,
        n_heads: int,
        ff_expansion: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=d_model,
        )
        self.pos_embed = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    n_heads,
                    ff_expansion,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

        # Weight tie
        self.lm_head = torch.nn.Linear(d_model, n_vocab, bias=False)
        self.lm_head.weight = self.embedding_layer.weight

        self.max_seq_len = max_seq_len
        self.d_model = d_model

    @torch.no_grad()
    def generate_with_cache(
        self, tokens: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation, no KV cache (recomputes the full
        sequence every step - O(n^2), fine for now, revisit as a later
        optimization pass).

        Args:
            tokens: (B, S) prompt token ids.
            max_new_tokens: how many tokens to generate after the prompt.
            eos_token_id: if any sequence in the batch generates this token, stop early.
        """
        self.eval()

        cache = [None] * len(self.layers)
        pos_in_sequence = 0
        for _ in range(max_new_tokens):
            if tokens.size(1) == self.max_seq_len:
                return tokens

            if cache[0] is not None:
                model_input = tokens[:, -1:]
            else:
                model_input = tokens

            logits, pos_in_sequence = self(model_input, cache, pos_in_sequence)
            next_token_logits = logits[
                :, -1, :
            ]  # Only need to worry about this during prefill

            # Since the output logits are essentially sim-search across the
            # embedding corpus, the next token ID is just argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tokens = torch.hstack([tokens, next_token])

            # TODO 6 (optional but recommended): if eos_token_id is set and
            # every sequence in the batch has produced it, break early instead
            # of running all max_new_tokens steps.

        return tokens

    @torch.no_grad()
    def generate(
        self, tokens: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation, no KV cache (recomputes the full
        sequence every step - O(n^2), fine for now, revisit as a later
        optimization pass).

        Args:
            tokens: (B, S) prompt token ids.
            max_new_tokens: how many tokens to generate after the prompt.
            eos_token_id: if any sequence in the batch generates this token, stop early.
        """
        self.eval()
        for _ in range(max_new_tokens):
            if tokens.size(1) == self.max_seq_len:
                return tokens

            logits, _ = self(tokens, None, 0)
            # You only care about the prediction for the *next* token,
            # which comes from the last position in the sequence dim.
            # Slice down (B, S, n_vocab) -> (B, n_vocab).
            next_token_logits = logits[:, -1, :]

            # Since the output logits are essentially sim-search across the
            # embedding corpus, the next token ID is just argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tokens = torch.hstack([tokens, next_token])

            # TODO 6 (optional but recommended): if eos_token_id is set and
            # every sequence in the batch has produced it, break early instead
            # of running all max_new_tokens steps.

        return tokens

    def forward(self, tokens, cache: list | None = None, pos_in_sequence: int = 0):
        x = self.embedding_layer(tokens)
        B, S, D = x.shape
        # scale input only since we are weight tying
        x = x / math.sqrt(D)

        if pos_in_sequence == 0:
            positions = torch.arange(pos_in_sequence, S, device=x.device)
            x = x + self.pos_embed(positions)
            pos_in_sequence = (
                S  # The index we're at after prefill, read in this token next
            )
        else:
            x = x + self.pos_embed(torch.tensor(pos_in_sequence, device=tokens.device))
            pos_in_sequence += 1

        use_cache = isinstance(cache, list)
        for i, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = cache[i]
                x, k, v = layer(x, layer_cache)
                cache[i] = [k, v]
            else:
                x, _, _ = layer(x, None)

        # we're weight tying here. On input, idx -> embedding lookup.
        # At this point (output), the mental model is "simlilarity search"
        # like you'd have in a vector db. embedding -> lm_head -> logits are
        # most probable next tokens
        # This ends up shape (B, S, n_vocab)
        return self.lm_head(self.norm(x)), pos_in_sequence
