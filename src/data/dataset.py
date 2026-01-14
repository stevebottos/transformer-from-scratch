"""Data pipeline for autoregressive language modeling."""

import os
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")


class TokenDataset(Dataset):
    """Fixed-length chunks for next-token prediction: x[i:i+n] -> y[i+1:i+n+1]."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + 1:idx + self.seq_len + 1]


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
    """
    Create train/val dataloaders from WikiText-103.

    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = tiktoken.get_encoding(encoding)
    data = load_wikitext()

    train_tokens = torch.tensor(tokenizer.encode(data["train"]), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(data["val"]), dtype=torch.long)

    train_loader = DataLoader(
        TokenDataset(train_tokens, seq_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        TokenDataset(val_tokens, seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, tokenizer
