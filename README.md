# Transformer From Scratch

A decoder-only transformer built up from scratch in PyTorch, trained on WikiText-103. Dense and Mixture-of-Experts variants, param-matched so the two are directly comparable.

## Architecture

- Pre-Norm (`x + Attn(Norm(x))`, `x + FFN(Norm(x))`), RMSNorm, SwiGLU feedforward
- `bias=False` on all linear projections, weight tying between embedding and lm_head
- GPT-2 (tiktoken) tokenizer
- Grouped-query attention (fewer KV heads than query heads) with a working KV cache for incremental decoding
- Mixture-of-Experts: router + per-expert FFN, Switch-style load-balancing aux loss, expert-capacity capping (overflow tokens dropped by lowest router confidence), grouped-GEMM dispatch (one batched matmul instead of a per-expert loop)
- Mixed precision + `torch.compile()` training, AdamW

## Layout

- `transformers_from_scratch/attention.py` — multi-head attention, GQA, KV cache
- `transformers_from_scratch/layers.py` — RMSNorm, SwiGLU, dense decoder layer; `NaiveMoEDecoderLayer` (masked-loop dispatch, kept as the readable reference) and `MoEDecoderLayer` (capacity-capped, grouped-GEMM dispatch)
- `transformers_from_scratch/transformer.py` — decoder-only model, dense or MoE
- `transformers_from_scratch/data/` — WikiText-103 loading and batching
- `train.py` / `utils.py` — training loop, mixed precision, checkpointing
- `tests/` — KV cache and GQA correctness (cached generation matches full recompute)

## Usage

```bash
uv sync

# Dense
uv run python train.py

# MoE
uv run python train.py --moe
```

Hyperparameters are arguments to `train()` in `train.py`.

## Tests

```bash
make test
```

## TODO

- [ ] Static-shaped MoE dispatch (scatter/gather instead of boolean masking) so `torch.compile` covers the grouped-GEMM path

## References

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [RMSNorm (2019)](https://arxiv.org/abs/1910.07467)
- [GLU Variants (2020)](https://arxiv.org/abs/2002.05202)
- [Switch Transformers (2021)](https://arxiv.org/abs/2101.03961)
- [Chinchilla Scaling Laws (2022)](https://arxiv.org/abs/2203.15556)
