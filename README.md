# Transformer From Scratch

A modern decoder-only transformer implementation in PyTorch, trained on WikiText-103.

## What's In / What's Out

Decoder-only, modern-style, but not chasing every LLaMA feature.

**In**
- Pre-Norm: `x + Attn(Norm(x))`, `x + FFN(Norm(x))`
- RMSNorm instead of LayerNorm
- SwiGLU feedforward
- `bias=False` on all linear projections (norm already gives a shift term; matches modern practice, not the 2017 paper)
- Weight tying between embedding and lm_head
- GPT-2 (tiktoken) tokenizer
- Mixed precision + `torch.compile()` training, AdamW

**Out (for now)**
- RoPE - skipped, using learned/absolute positions instead. Revisit if extrapolation to longer contexts matters.
- KV cache / efficient inference - see TODO
- Encoder / cross-attention - see TODO

## Scaling Laws

WikiText-103 has ~100M tokens. Chinchilla-optimal (tokens/params ≈ 20) suggests ~5M params, but GPT-2's 50k vocab means embeddings alone exceed this. Current setup is overparameterized - use regularization accordingly.

Run `python scaling_laws.py` to see suggested configurations.

## Usage

```bash
# Install dependencies
pip install torch einops tiktoken datasets tqdm

# Train
python train.py

# Adjust hyperparameters in train.py:
train(
    d_model=256,
    n_layers=6,
    n_heads=4,
    seq_len=128,
    batch_size=64,
    lr=3e-4,
    dropout=0.1,
)
```

## TODO

- [ ] **Step 6: Inference & KV Cache**
  - Greedy/sampling decoding loop
  - KV caching for efficient autoregressive generation
  - Generation speed benchmarks

- [ ] **Step 7: Encoder & Cross-Attention**
  - CrossAttention module
  - EncoderBlock and Encoder stack
  - EncoderDecoderTransformer

- [ ] **Step 8: Seq2Seq Training**
  - Toy task dataset (reversal, copy)
  - Train encoder-decoder model

## References

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: RoPE (2021)](https://arxiv.org/abs/2104.09864)
- [RMSNorm (2019)](https://arxiv.org/abs/1910.07467)
- [GLU Variants (2020)](https://arxiv.org/abs/2002.05202)
- [Chinchilla Scaling Laws (2022)](https://arxiv.org/abs/2203.15556)
