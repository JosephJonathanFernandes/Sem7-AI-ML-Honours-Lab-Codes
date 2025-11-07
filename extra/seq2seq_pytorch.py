"""
Seq2Seq (Encoder–Decoder RNN) — PyTorch

Minimal example with teacher forcing on synthetic token sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)

    def forward(self, src):
        x = self.emb(src)          # (N, T_src, E)
        out, (h, c) = self.lstm(x) # out: (N, T_src, H)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, tgt, h, c):
        # teacher forcing: feed entire target sequence (tokens)
        x = self.emb(tgt)          # (N, T_tgt, E)
        out, (h, c) = self.lstm(x, (h, c))
        logits = self.fc(out)      # (N, T_tgt, V)
        return logits, (h, c)


def main() -> None:
    torch.manual_seed(0)

    # Synthetic token data
    N, T_src, T_tgt, V = 64, 6, 6, 8
    src = torch.randint(0, V, (N, T_src))
    tgt = torch.randint(0, V, (N, T_tgt))

    enc = Encoder(vocab_size=V, emb=16, hidden=64)
    dec = Decoder(vocab_size=V, emb=16, hidden=64)

    params = list(enc.parameters()) + list(dec.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train a few epochs
    for epoch in range(5):
        optim.zero_grad()
        h, c = enc(src)
        logits, _ = dec(tgt, h, c)
        # Flatten for CE: (N*T_tgt, V) vs targets (N*T_tgt)
        loss = criterion(logits.reshape(-1, V), tgt.reshape(-1))
        loss.backward()
        optim.step()
        print(f"Epoch {epoch+1}/5 - loss={loss.item():.4f}")

    # Inference (greedy) for one example
    enc.eval(); dec.eval()
    with torch.no_grad():
        h, c = enc(src[:1])
        # start from a random token; ordinarily use a BOS token
        token = torch.randint(0, V, (1, 1))
        out_tokens = []
        for _ in range(T_tgt):
            logits, (h, c) = dec(token, h, c)  # feeding last step only
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out_tokens.append(int(next_token.item()))
            token = next_token
        print("Greedy decoded tokens:", out_tokens)


if __name__ == "__main__":
    main()
