"""
Step 3: Layers & Normalization

Implement the following:
1. LayerNormalization:
    - Implement from scratch: (x - mean) / (std + eps) * gamma + beta.
2. PositionwiseFeedForward:
    - Two linear layers with a non-linearity (ReLU or GELU) in between.
    - Typically d_ff = 4 * d_model.
3. DecoderBlock:
    - Self-attention layer.
    - Add & Norm (residual connection followed by LayerNorm).
    - Feed-forward layer.
    - Add & Norm.
"""
import torch
import torch.nn as nn

# Implement classes here
