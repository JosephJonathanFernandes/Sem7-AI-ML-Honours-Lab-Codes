import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

# Dataset: 3 binary patterns (A, B, C)
patterns = np.array([
    [1, -1, 1, -1,
     -1, 1, -1, 1,
      1, -1, 1, -1,
     -1, -1, 1, -1],   # Pattern A

    [1, 1, -1, -1,
     -1, 1, -1, 1,
      1, 1, -1, -1,
      1, 1, -1, -1],   # Pattern B

    [1, -1, -1, -1,
     -1, 1, -1, -1,
      1, -1, -1, -1,
      1, 1, -1, -1]    # Pattern C
])

n = patterns.shape[1]
W = np.zeros((n, n))

# Training (Hebbian learning)
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

# Test with a noisy version of Pattern A
test = np.array([1, -1, -1, -1,
                 -1, 1, -1, 1,
                  1, -1, -1, -1,
                 -1, -1, 1, -1])

print("Noisy Input:\n", test.reshape(4,4))

# Recall process
for _ in range(5):
    test = sign(np.dot(W, test))

print("\nRecalled Pattern:\n", test.reshape(4,4))
