import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

# Define patterns
patterns = np.array([[1, -1, 1, -1],
                     [-1, 1, -1, 1]])

# Initialize weight matrix
W = np.zeros((4, 4))

# Training phase
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)   # No self-connection

# Test with a noisy input
test = np.array([1, -1, -1, -1])

# Recall process
for _ in range(5):
    test = sign(np.dot(W, test))

print("Recalled Pattern:", test)
