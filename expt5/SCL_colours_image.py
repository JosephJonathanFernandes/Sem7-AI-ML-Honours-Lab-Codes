import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

# Load a real-world image (example: "china.jpg" from sklearn)
image = load_sample_image("china.jpg")
image = np.array(image, dtype=np.float64) / 255  # normalize

# Flatten image data to (n_samples, 3)
data = np.reshape(image, (-1, 3))
data = shuffle(data, random_state=42)[:10000]  # sample pixels for training

# Number of clusters (dominant colors)
clusters = 6
weights = np.random.rand(clusters, 3)

# Hyperparameters
lr = 0.3
epochs = 60

# Training
for _ in range(epochs):
    for x in data:
        winner = np.argmin(np.linalg.norm(x - weights, axis=1))
        weights[winner] += lr * (x - weights[winner])
    lr *= 0.95  # decay learning rate

# Assign each pixel to its nearest cluster
flat_image = np.reshape(image, (-1, 3))
labels = np.argmin(np.linalg.norm(flat_image[:, None] - weights[None, :], axis=2), axis=1)
compressed_img = weights[labels].reshape(image.shape)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow([weights])
plt.title("Dominant Colors (SCL)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(compressed_img)
plt.title("Reconstructed Image (Clustered)")
plt.axis("off")

plt.tight_layout()
plt.show()
