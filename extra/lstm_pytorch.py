"""
LSTM (Long Short-Term Memory) — PyTorch

Dummy binary classification on synthetic sequential data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    # Synthetic data: (samples, timesteps, features)
    N, T, F_in = 100, 10, 8
    X = torch.randn(N, T, F_in)
    y = torch.randint(0, 2, (N,), dtype=torch.float32)

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size: int, hidden: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)  # (N, T, H)
            last = out[:, -1, :]   # (N, H)
            logits = self.fc(last) # (N, 1)
            return logits.squeeze(1)

    model = LSTMClassifier(F_in, 32)
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

    print("✅ LSTM Model Trained")


if __name__ == "__main__":
    main()
