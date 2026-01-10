"""
Step 2: Attention Mechanisms

Implement the following:
1. MultiHeadAttention:
    - Scaled Dot-Product Attention (manual implementation).
    - Multi-head splitting and concatenation.
    - Linear projections for Q, K, and V.
    - Masking support (optional mask tensor).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Implement classes here
