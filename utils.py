"""Shared helpers for train.py and train_distributed.py."""

import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import autocast

from transformers_from_scratch.transformer import DecoderOnlyTransformer

CHECKPOINT_DIR = Path("checkpoints")

# Same backbone (d_model/n_layers/n_heads) and same per-expert FFN size as
# dense's ff_expansion, so active compute/token is identical between the two.
# MoE's extra experts add total capacity (params) for free at inference time,
# which is the whole point of MoE - not something to cancel out.
DENSE_CONFIG = dict(ff_expansion=4, n_experts=None)
MOE_CONFIG = dict(ff_expansion=4, n_experts=8)


def build_model(
    d_model, n_vocab, n_layers, n_heads, dropout, seq_len, moe, device
):
    """Construct the dense or MoE decoder-only transformer."""
    model_config = MOE_CONFIG if moe else DENSE_CONFIG
    return DecoderOnlyTransformer(
        d_model=d_model,
        n_vocab=n_vocab,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=seq_len,
        **model_config,
    ).to(device)


def save_checkpoint(model, optimizer, scaler, step, val_loss, config, ckpt_path):
    """Save model checkpoint."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": config,
        },
        ckpt_path,
    )


@torch.no_grad()
def generate_sample(
    model, val_loader, tokenizer, device, prompt_len=10, max_new_tokens=30
):
    """Greedily generate from a validation prompt, for eyeballing progress."""
    model.eval()
    x, _ = next(iter(val_loader))
    prompt = x[:1, :prompt_len].to(device)
    out = model.generate(prompt, max_new_tokens=max_new_tokens)
    model.train()
    return tokenizer.decode(out[0].tolist())


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    """Average validation loss over at most max_batches batches."""
    model.eval()
    total_loss = 0
    n_batches = 0
    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast("cuda"):
            logits, _, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches
