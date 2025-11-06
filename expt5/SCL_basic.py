import numpy as np
import matplotlib.pyplot as plt

# Generate random 2D data points
data = np.random.rand(100, 2)

# Number of clusters (neurons)
clusters = 3
weights = np.random.rand(clusters, 2)

# Learning parameters
lr = 0.2
epochs = 100

# Training
for _ in range(epochs):
    for x in data:
        # Find the winning neuron (closest weight)
        winner = np.argmin(np.linalg.norm(x - weights, axis=1))
        # Update only the winner's weights
        weights[winner] += lr * (x - weights[winner])
    lr *= 0.99  # decay learning rate

# Visualization
colors = ['r', 'g', 'b']
for i in range(clusters):
    plt.scatter(weights[i][0], weights[i][1], c=colors[i], marker='X', s=200, label=f'Neuron {i+1}')

plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.6, label='Data Points')
plt.title("Clustering using Self-Organizing Competitive Learning")
plt.legend()
plt.show()
