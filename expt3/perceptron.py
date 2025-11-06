"""
Title: Perceptron Learning Algorithm – Interactive Visualization
Author: Joseph Jonathan Fernandes
Description:
Implements a Perceptron for binary classification and dynamically visualizes 
its decision boundary evolution over epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------- Dataset ---------------------
# Linearly separable data (2D)
X = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Two classes

# --------------------- Perceptron Class ---------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.history = []

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                z = np.dot(X[i], self.w) + self.b
                y_pred = self.activation(z)
                update = self.lr * (y[i] - y_pred)
                self.w += update * X[i]
                self.b += update
            self.history.append((self.w.copy(), self.b))

# --------------------- Training ---------------------
model = Perceptron(lr=0.2, epochs=15)
model.fit(X, y)

# --------------------- Animation Setup ---------------------
fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_title("Perceptron Learning – Decision Boundary Evolution")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=100)
line, = ax.plot([], [], 'k--')

def update(frame):
    w, b = model.history[frame]
    x_vals = np.array([0, 5])
    y_vals = -(b + w[0]*x_vals)/w[1]
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {frame+1} | Weights: {np.round(w,2)}, Bias: {b:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=len(model.history), repeat=False)
plt.show()
