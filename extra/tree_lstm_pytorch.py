"""
Tree-LSTM (Binary) — PyTorch

Defines a simple binary Tree-LSTM cell and composes a tiny tree.
"""

import torch
import torch.nn as nn


class BinaryTreeLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # i, o, u from x and [h_l, h_r]
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(2 * hidden_size, 3 * hidden_size)
        # separate forget gates for left and right child
        self.W_f_l = nn.Linear(input_size, hidden_size)
        self.U_f_l = nn.Linear(2 * hidden_size, hidden_size)
        self.W_f_r = nn.Linear(input_size, hidden_size)
        self.U_f_r = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, h_l, c_l, h_r, c_r):
        h_cat = torch.cat([h_l, h_r], dim=1)
        iou = self.W_iou(x) + self.U_iou(h_cat)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        f_l = torch.sigmoid(self.W_f_l(x) + self.U_f_l(h_cat))
        f_r = torch.sigmoid(self.W_f_r(x) + self.U_f_r(h_cat))

        c = i * u + f_l * c_l + f_r * c_r
        h = o * torch.tanh(c)
        return h, c


def main() -> None:
    torch.manual_seed(0)
    input_size, hidden = 5, 8
    cell = BinaryTreeLSTMCell(input_size, hidden)

    # Create three leaves (x1, x2, x3)
    x1 = torch.randn(1, input_size)
    x2 = torch.randn(1, input_size)
    x3 = torch.randn(1, input_size)

    # Leaf states: children are zeros
    h0 = torch.zeros(1, hidden)
    c0 = torch.zeros(1, hidden)

    # Compose (x1, x2) -> p12
    h12, c12 = cell(x=torch.randn(1, input_size), h_l=h0, c_l=c0, h_r=h0, c_r=c0)
    # Compose (p12, x3) -> root
    h_root, c_root = cell(x=torch.randn(1, input_size), h_l=h12, c_l=c12, h_r=h0, c_r=c0)

    print("Tree-LSTM root hidden state shape:", h_root.shape)


if __name__ == "__main__":
    main()
