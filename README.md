# Transformer From Scratch

A modern decoder-only transformer implementation in PyTorch, trained on WikiText-103.

## What's Implemented

### Model Architecture (`src/models/`)

**Embeddings** (`embeddings.py`)
- Token embeddings without scaling (scaling unnecessary with RoPE, simplifies weight tying)

**Attention** (`attention.py`)
- Multi-head attention with `einops` for tensor manipulation
- RoPE (Rotary Positional Embeddings) using real-valued sin/cos (compile-friendly, no complex numbers)
- Switchable backends: manual SDPA or PyTorch's Flash Attention (`use_torch_sdpa=True`)
- All projections `bias=False`

**Layers** (`layers.py`)
- RMSNorm (more efficient than LayerNorm)
- SwiGLU feedforward (gated activation, 3 linear layers)
- DecoderBlock with Pre-Norm architecture: `x + Attn(Norm(x))` then `x + FFN(Norm(x))`

**Transformer** (`transformer.py`)
- DecoderOnlyTransformer assembling all components
- Precomputed RoPE frequencies as buffers
- Weight tying between embedding and lm_head
- Xavier init for linears, normal(0, 0.02) for embeddings

### Training (`train.py`)

- Mixed precision training (autocast + GradScaler)
- AdamW optimizer with weight decay
- `torch.compile()` for speed
- Flash Attention verification
- Checkpointing (saves latest to `checkpoints/latest.pt`)
- tqdm progress bars

### Data (`src/data/`)

- WikiText-103 from HuggingFace (~100M tokens)
- tiktoken (GPT-2) tokenizer
- Efficient DataLoader with `num_workers`, `pin_memory`, `persistent_workers`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No embedding scaling | Original paper scaled by √d_model for sinusoidal pos encodings. RoPE doesn't add to embeddings, so unnecessary. Also simplifies weight tying. |
| Real-valued RoPE | Complex number ops break `torch.compile()`. Real sin/cos is mathematically identical and compiles. |
| Pre-Norm | Better gradient flow than Post-Norm, more stable training for deep networks. |
| `is_causal=True` | More efficient than explicit mask with Flash Attention - uses fused kernel. |
| No dropout by default | LLaMA-style. Dropout available via parameter if needed. |
| GPT-2 tokenizer | Large vocab (50k) but well-tested. Trade-off: embedding dominates params for small models. |

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
