"""
RNTN (Recursive Neural Tensor Network) — NumPy

Adds a bilinear tensor interaction term to the Tree-RNN composition.
"""

import numpy as np


def rntn(node1: np.ndarray, node2: np.ndarray, W: np.ndarray, V: np.ndarray, b: np.ndarray) -> np.ndarray:
    # interaction: V (H x H x H) contracted with outer(node1, node2)
    interaction = np.tensordot(V, np.outer(node1, node2), axes=([1, 2], [0, 1]))
    return np.tanh(W @ np.concatenate([node1, node2]) + interaction + b)


def main() -> None:
    H = 3
    node1 = np.random.randn(H)
    node2 = np.random.randn(H)
    W = np.random.randn(H, 2 * H)
    V = np.random.randn(H, H, H)
    b = np.random.randn(H)

    parent = rntn(node1, node2, W, V, b)
    print("RNTN parent vector:", parent)


if __name__ == "__main__":
    main()
