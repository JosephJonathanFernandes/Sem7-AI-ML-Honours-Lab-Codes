"""
Tree-RNN (Recursive Neural Network) — NumPy

Combines two child vectors into a parent using a tanh nonlinearity.
"""

import numpy as np


def combine(node1: np.ndarray, node2: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.tanh(W @ np.concatenate([node1, node2]) + b)


def main() -> None:
    # Example word embeddings
    word_vec = {
        "good": np.array([0.8, 0.6], dtype=float),
        "movie": np.array([0.4, 0.9], dtype=float),
    }

    W = np.random.randn(2, 4)
    b = np.random.randn(2)

    vec_parent = combine(word_vec["good"], word_vec["movie"], W, b)
    print("Tree-RNN parent representation:", vec_parent)


if __name__ == "__main__":
    main()
