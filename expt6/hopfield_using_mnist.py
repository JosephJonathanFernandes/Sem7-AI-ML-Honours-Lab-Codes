import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

def sign(x):
    return np.where(x >= 0, 1, -1)

# 1️⃣ Load MNIST dataset (small subset for speed)
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# Select digits 0, 1, 2 for training
selected_digits = [0, 1, 2]
idx = np.isin(y, selected_digits)
X, y = X[idx], y[idx]

# 2️⃣ Normalize and reduce dimensionality
X = X / 255.0
pca = PCA(n_components=100)  # reduce from 784 → 100
X_pca = pca.fit_transform(X)

# 3️⃣ Convert to binary patterns (+1 / -1)
patterns = np.where(X_pca > 0, 1, -1)

# Take first 3 samples (one of each digit)
samples = []
for digit in selected_digits:
    samples.append(patterns[np.where(y == digit)[0][0]])
samples = np.array(samples)

# 4️⃣ Train Hopfield network
n = samples.shape[1]
W = np.zeros((n, n))
for p in samples:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

# 5️⃣ Add noise to one pattern
test = samples[0].copy()
noise_idx = np.random.choice(len(test), size=15, replace=False)
test[noise_idx] *= -1

# 6️⃣ Recall process
for _ in range(5):
    test = sign(np.dot(W, test))

# 7️⃣ Reconstruct recalled image (inverse PCA)
reconstructed_original = pca.inverse_transform(samples[0])
reconstructed_noisy = pca.inverse_transform(samples[0] * np.random.choice([1, -1], size=len(samples[0]), p=[0.9, 0.1]))
reconstructed_recalled = pca.inverse_transform(test)

# 8️⃣ Visualization
fig, ax = plt.subplots(1, 3, figsize=(8, 3))
ax[0].imshow(reconstructed_original.reshape(28, 28), cmap='gray')
ax[0].set_title("Original Pattern")

ax[1].imshow(reconstructed_noisy.reshape(28, 28), cmap='gray')
ax[1].set_title("Noisy Input")

ax[2].imshow(reconstructed_recalled.reshape(28, 28), cmap='gray')
ax[2].set_title("Recalled Pattern")

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
