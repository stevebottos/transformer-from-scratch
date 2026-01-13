"""
Step 5: Training Loop

Train the decoder-only transformer on WikiText-103 with mixed precision.
"""

import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.data import create_dataloaders
from src.models.transformer import DecoderOnlyTransformer

CHECKPOINT_DIR = Path("checkpoints")


def save_checkpoint(model, optimizer, scaler, epoch, step, val_loss, config):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "config": config,
        },
        CHECKPOINT_DIR / "latest.pt",
    )


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with autocast("cuda"):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        total_loss += loss.item()
    return total_loss / len(val_loader)


def train(
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 8,
    seq_len: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    epochs: int = 10,
    dropout: float = 0.3,
    eval_interval: int = 50000,
    device: str = "cuda",
):
    config = {k: v for k, v in locals().items()}
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, tokenizer = create_dataloaders(
        seq_len=seq_len, batch_size=batch_size
    )

    # Model
    model = DecoderOnlyTransformer(
        d_model=d_model,
        n_vocab=tokenizer.n_vocab,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=seq_len,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify Flash Attention is available
    x_test = torch.randint(0, tokenizer.n_vocab, (2, seq_len), device=device)
    try:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            with autocast("cuda"):
                _ = model(x_test)
        print("Flash Attention: enabled")
    except RuntimeError as e:
        print(f"Flash Attention: not available ({e})")

    model = torch.compile(model)

    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda")

    # Training
    step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % eval_interval == 0 and step > 0:
                val_loss = evaluate(model, val_loader, device)
                pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scaler, epoch, step, val_loss, config
                    )

            step += 1

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f}")
        save_checkpoint(model, optimizer, scaler, epoch, step, val_loss, config)

    return model, tokenizer


if __name__ == "__main__":
    train()
