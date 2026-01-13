"""
Step 5: Data Pipeline

Dataset for autoregressive language modeling using tiktoken.
Supports WikiText-103 (default) and custom text.
"""

import os

import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")


class TokenDataset(Dataset):
    """
    Autoregressive dataset: given tokens[i:i+seq_len], predict tokens[i+1:i+seq_len+1].
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_wikitext(version: str = "wikitext-103-raw-v1") -> dict[str, str]:
    """Load WikiText dataset from HuggingFace."""
    ds = load_dataset("wikitext", version, token=HF_TOKEN)
    return {
        "train": "\n".join(ds["train"]["text"]),
        "val": "\n".join(ds["validation"]["text"]),
        "test": "\n".join(ds["test"]["text"]),
    }


def create_dataloaders(
    seq_len: int,
    batch_size: int,
    encoding: str = "gpt2",
) -> tuple[DataLoader, DataLoader, tiktoken.Encoding]:
    """Create train/val dataloaders from WikiText-103."""
    tokenizer = tiktoken.get_encoding(encoding)

    data = load_wikitext()
    train_tokens = torch.tensor(tokenizer.encode(data["train"]), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(data["val"]), dtype=torch.long)

    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    train_dataset = TokenDataset(train_tokens, seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tokenizer = create_dataloaders(seq_len=64, batch_size=4)

    print(f"Vocab size: {tokenizer.n_vocab}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    x, y = next(iter(train_loader))
    print(f"Shapes: x={x.shape}, y={y.shape}")
    print(f"Input:  {repr(tokenizer.decode(x[0].tolist()))}")
    print(f"Target: {repr(tokenizer.decode(y[0].tolist()))}")
