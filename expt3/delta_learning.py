"""
Title: Delta Learning Rule (Widrow-Hoff) – Real-Time MSE Visualization
Author: Joseph Jonathan Fernandes
Description:
Implements the delta rule for continuous learning and visualizes 
the Mean Squared Error (MSE) reduction dynamically.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------- Dataset ---------------------
# XOR-like data (non-linear, continuous output learning)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])   # Expected continuous outputs

# --------------------- Delta Rule Model ---------------------
class DeltaRule:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.uniform(-1, 1, X.shape[1])
        self.b = np.random.uniform(-1, 1)
        self.mse_history = []

    def fit(self, X, y):
        for _ in range(self.epochs):
            net = np.dot(X, self.w) + self.b
            y_pred = 1 / (1 + np.exp(-net))  # Sigmoid activation
            error = y - y_pred
            self.w += self.lr * X.T.dot(error * y_pred * (1 - y_pred))
            self.b += self.lr * np.sum(error * y_pred * (1 - y_pred))
            mse = np.mean(error**2)
            self.mse_history.append(mse)

    def predict(self, X):
        net = np.dot(X, self.w) + self.b
        return 1 / (1 + np.exp(-net))

# --------------------- Training ---------------------
model = DeltaRule(lr=0.5, epochs=100)
model.fit(X, y)

# --------------------- Visualization ---------------------
plt.figure(figsize=(8,4))
plt.plot(model.mse_history, color='magenta', lw=2)
plt.title("Delta Learning Rule – MSE Convergence")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()

# --------------------- Results ---------------------
print("Final Weights:", np.round(model.w, 3))
print("Final Bias:", round(model.b, 3))
print("Predictions:", np.round(model.predict(X), 3))
