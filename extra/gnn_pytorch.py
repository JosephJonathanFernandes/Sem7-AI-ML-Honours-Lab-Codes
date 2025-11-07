"""
Simple Graph Neural Network — PyTorch

Two message-passing layers using adjacency multiplication.
"""

import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, X, A):
        H = torch.relu(self.fc1(A @ X))
        H = self.fc2(A @ H)
        return H


def main() -> None:
    torch.manual_seed(0)
    # Example: 4-node graph
    A = torch.tensor([[1, 1, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [0, 0, 1, 1]], dtype=torch.float32)
    X = torch.randn(4, 5)
    gnn = SimpleGNN(5, 3)
    output = gnn(X, A)
    print("GNN output node embeddings:\n", output)


if __name__ == "__main__":
    main()
