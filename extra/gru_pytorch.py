"""
GRU (Gated Recurrent Unit) — PyTorch

Dummy binary classification on synthetic sequential data.
"""

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(0)

    N, T, F_in = 100, 10, 8
    X = torch.randn(N, T, F_in)
    y = torch.randint(0, 2, (N,), dtype=torch.float32)

    class GRUClassifier(nn.Module):
        def __init__(self, input_size: int, hidden: int):
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            last = out[:, -1, :]
            logits = self.fc(last)
            return logits.squeeze(1)

    model = GRUClassifier(F_in, 32)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        optim.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch+1}/5 - loss={loss.item():.4f} acc={acc:.3f}")

    print("✅ GRU Model Trained")


if __name__ == "__main__":
    main()
