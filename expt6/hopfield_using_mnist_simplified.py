# Hopfield Network on MNIST (Simplified Fast Version)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

def sign(x):
    return np.where(x >= 0, 1, -1)

print("Loading MNIST subset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# ✅ Use only digits 0, 1, and 2 (first 1 sample each)
selected_digits = [0, 1, 2]
samples = []
for digit in selected_digits:
    idx = np.where(y == digit)[0][0]
    img = X[idx] / 255.0
    samples.append(img)
samples = np.array(samples)

# ✅ Dimensionality reduction (fast)
pca = PCA(n_components=64)
patterns = pca.fit_transform(samples)
patterns = np.where(patterns > 0, 1, -1)  # binarize

# ✅ Train Hopfield Network
n = patterns.shape[1]
W = np.zeros((n, n))
for p in patterns:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)

# ✅ Pick one pattern (e.g., digit 0) and add noise
test = patterns[0].copy()
noise_idx = np.random.choice(n, size=8, replace=False)
test[noise_idx] *= -1

# ✅ Recall
for _ in range(4):
    test = sign(np.dot(W, test))

# ✅ Reconstruct images
orig_img = pca.inverse_transform(patterns[0])
noisy_img = pca.inverse_transform(np.where(patterns[0] == test, patterns[0], -patterns[0]))
recalled_img = pca.inverse_transform(test)

# ✅ Display results
fig, ax = plt.subplots(1, 3, figsize=(8, 3))
ax[0].imshow(orig_img.reshape(28, 28), cmap='gray')
ax[0].set_title("Original Digit")

ax[1].imshow(noisy_img.reshape(28, 28), cmap='gray')
ax[1].set_title("Noisy Input")

ax[2].imshow(recalled_img.reshape(28, 28), cmap='gray')
ax[2].set_title("Recalled Digit")

for a in ax: a.axis('off')
plt.tight_layout()
plt.show()
