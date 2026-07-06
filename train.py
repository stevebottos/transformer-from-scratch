"""Training loop for decoder-only transformer on WikiText-103."""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from transformers_from_scratch.data import create_dataloaders
from utils import CHECKPOINT_DIR, build_model, generate_sample, save_checkpoint


def train(
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    seq_len: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    max_steps: int = 500000,
    dropout: float = 0.1,
    eval_interval: int = 5000,
    device: str = "cuda",
    moe: bool = False,
    aux_loss_weight: float = 0.01,
):
    """Train the model."""
    config = dict(locals())
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, tokenizer = create_dataloaders(seq_len, batch_size)

    model = build_model(
        d_model, tokenizer.n_vocab, n_layers, n_heads, dropout, seq_len, moe, device
    )

    print(
        f"Device: {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda")

    start_step = 0
    best_val_loss = float("inf")
    val_loss = 0
    ckpt_path = CHECKPOINT_DIR / ("latest_moe.pt" if moe else "latest_dense.pt")
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        best_val_loss = val_loss = ckpt["val_loss"]

    # Compile after loading weights - the compiled wrapper's state_dict keys
    # don't cleanly match the raw module's, easier to load into the raw
    # module first.
    model = torch.compile(model)

    model.train()
    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")
    train_iter = iter(train_loader)
    for step in range(start_step, max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            logits, _, aux_loss = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss_weight * aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)

        # if step % eval_interval == 0 and step > 0:
        #     val_loss = evaluate(model, val_loader, device)
        #     model.train()
        #     pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}")
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         save_checkpoint(model, optimizer, scaler, step, val_loss, config)

        if step % 1000 == 0 and step > 0:
            sample = generate_sample(model, val_loader, tokenizer, device)
            tqdm.write(f"[step {step}] sample: {sample!r}")

            save_checkpoint(
                model, optimizer, scaler, max_steps, val_loss, config, ckpt_path
            )

    # val_loss = evaluate(model, val_loader, device)
    print(f"Final | Val Loss: {val_loss:.4f}")
    save_checkpoint(model, optimizer, scaler, max_steps, val_loss, config)

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--moe", action="store_true")
    args = parser.parse_args()
    train(moe=args.moe)
