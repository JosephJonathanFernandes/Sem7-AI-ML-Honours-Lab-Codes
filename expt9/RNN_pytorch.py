"""
PyTorch version of expt9/RNN.py

Trains a tiny character-level RNN on the string "hello world" to predict the next
character given a 3-character window, matching the Keras example's behavior.

How to run (Windows PowerShell):
    python .\expt9\RNN_pytorch.py

Requirements:
    - torch

Notes:
    - Uses one-hot inputs to mirror the Keras setup (input_size = vocab_size).
    - Runs on CPU by default; no GPU needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CharData:
    text: str
    chars: List[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]
    X_onehot: torch.Tensor  # shape: (N, T, V)
    y_idx: torch.Tensor     # shape: (N,)


def build_dataset(text: str, seq_len: int = 3) -> CharData:
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    X_idx: List[List[int]] = []
    y_idx: List[int] = []
    for i in range(len(text) - seq_len):
        seq = text[i : i + seq_len]
        target = text[i + seq_len]
        X_idx.append([char_to_idx[c] for c in seq])
        y_idx.append(char_to_idx[target])

    X_idx_np = np.array(X_idx, dtype=np.int64)
    y_idx_np = np.array(y_idx, dtype=np.int64)

    V = len(chars)
    N, T = X_idx_np.shape

    # One-hot encode to mirror Keras example (input_size = V)
    X_onehot_np = np.eye(V, dtype=np.float32)[X_idx_np]  # (N, T, V)

    X_onehot = torch.from_numpy(X_onehot_np)  # float32
    y_idx_t = torch.from_numpy(y_idx_np)      # int64

    return CharData(
        text=text,
        chars=chars,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        X_onehot=X_onehot,
        y_idx=y_idx_t,
    )


class TinyCharRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, V) one-hot
        out, h_n = self.rnn(x)  # out: (N, T, H)
        last = out[:, -1, :]    # (N, H)
        logits = self.fc(last)  # (N, V)
        return logits


def train(model: nn.Module, data: CharData, epochs: int = 200, lr: float = 1e-2) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X = data.X_onehot.to(device)
    y = data.y_idx.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Optional small progress print every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | acc={acc:.3f}")


@torch.no_grad()
def predict_next_chars(model: nn.Module, data: CharData) -> List[str]:
    device = next(model.parameters()).device
    model.eval()

    X = data.X_onehot.to(device)
    logits = model(X)
    pred_idx = logits.argmax(dim=1).cpu().numpy().tolist()
    return [data.idx_to_char[i] for i in pred_idx]


def main():
    # Match the original dataset
    text = "hello world"
    seq_len = 3
    data = build_dataset(text, seq_len)

    V = len(data.chars)
    hidden_size = 32
    model = TinyCharRNN(input_size=V, hidden_size=hidden_size, num_classes=V)

    train(model, data, epochs=200, lr=1e-2)

    preds = predict_next_chars(model, data)
    for i, p in enumerate(preds):
        seq = text[i : i + seq_len]
        print(f"{seq} -> {p}")


if __name__ == "__main__":
    main()
