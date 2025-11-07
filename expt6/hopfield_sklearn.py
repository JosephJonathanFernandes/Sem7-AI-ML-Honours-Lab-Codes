import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def sign(x):
    return np.where(x >= 0, 1, -1)

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Pick 3 digits to store (e.g., 0, 1, 2)
selected = [0, 1, 2]
patterns = []

for digit in selected:
    idx = np.where(y == digit)[0][0]
    img = X[idx]
    img = np.where(img > 8, 1, -1)  # Binarize
    patterns.append(img)

patterns = np.array(patterns)
n = patterns.shape[1]
W = np.zeros((n, n))

# Train using Hebbian rule
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

# Choose one pattern and add noise
test = patterns[0].copy()
noise_idx = np.random.choice(len(test), size=10, replace=False)
test[noise_idx] *= -1  # Flip bits (add noise)

# Recall process
for _ in range(5):
    test = sign(np.dot(W, test))

# Display results
fig, ax = plt.subplots(1, 2, figsize=(5, 3))
ax[0].imshow(patterns[0].reshape(8, 8), cmap='gray')
ax[0].set_title("Original Digit 0")

ax[1].imshow(test.reshape(8, 8), cmap='gray')
ax[1].set_title("Recalled Digit")

plt.show()
