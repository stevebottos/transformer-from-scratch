"""
Correctness proof for the KV cache: cached, autoregressive generation must
agree with a single-shot, no-cache forward pass over the same final
sequence. At every generated position, the token that was actually chosen
during cached generation must be exactly what a full recompute would have
predicted as the argmax next token from that position. A mismatch means a
bug in the cache wiring (stale k/v, wrong position id, mask applied when it
shouldn't be, etc) - not a training/quality issue, since weights are
untrained and fixed either way.
"""

import torch

from transformers_from_scratch.transformer import DecoderOnlyTransformer

VOCAB_SIZE = 100
MAX_SEQ_LEN = 16


def make_model():
    torch.manual_seed(0)
    return DecoderOnlyTransformer(
        d_model=32,
        n_vocab=VOCAB_SIZE,
        n_layers=2,
        n_heads=4,
        max_seq_len=MAX_SEQ_LEN,
    )


def test_cached_generation_matches_full_recompute():
    model = make_model()
    prompt = torch.randint(0, VOCAB_SIZE, (2, 5))
    prompt_len = prompt.shape[1]

    out = model.generate(prompt.clone(), max_new_tokens=6)

    # Ground truth: one no-cache forward pass over the whole final sequence.
    logits, _, _ = model.forward(out)

    # Every generated token should be exactly the argmax next-token
    # prediction a full recompute would have made from the position it
    # was generated at.
    predicted_next = logits[:, prompt_len - 1 : -1, :].argmax(dim=-1)
    actual_next = out[:, prompt_len:]

    assert torch.equal(predicted_next, actual_next), (
        "cached generation diverged from a full no-cache recompute:\n"
        f"predicted: {predicted_next}\n"
        f"actual:    {actual_next}"
    )
