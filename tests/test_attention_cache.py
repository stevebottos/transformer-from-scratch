"""
Isolated correctness proof for the KV cache, one level below generation:
does MultiHeadAttention with a growing cache produce the same output as
attending over the full sequence at once, with no generation loop, no
position-id plumbing, nothing else in the way.
"""

import torch

from transformers_from_scratch.attention import MultiHeadAttention

D_MODEL = 32
N_HEADS = 4
SEQ_LEN = 6
BATCH = 2


def test_stepwise_cache_matches_full_sequence_attention():
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS).eval()

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
    out_full, _, _ = mha(x, mask=causal_mask, cache=None)

    cache = None
    out_steps = []
    for t in range(SEQ_LEN):
        x_t = x[:, t : t + 1, :]
        out_t, k, v = mha(x_t, mask=None, cache=cache)
        cache = (k, v)
        out_steps.append(out_t)
    out_stepwise = torch.cat(out_steps, dim=1)

    assert torch.allclose(out_full, out_stepwise, atol=1e-5), (
        "stepwise cached attention diverged from full-sequence attention:\n"
        f"full:     {out_full}\n"
        f"stepwise: {out_stepwise}"
    )
