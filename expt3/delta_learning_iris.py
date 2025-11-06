"""
Title: Delta Learning Rule (Widrow-Hoff) using Iris Dataset
Author: Joseph Jonathan Fernandes
Description:
Implements the Delta Learning Rule (using sigmoid activation)
for classifying Iris Setosa vs Versicolor and visualizes MSE convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------- Load and Prepare Dataset ---------------------
iris = load_iris()
X = iris.data[:100, :2]   # Two features
y = iris.target[:100].astype(float)

# Normalize input features
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Delta Rule Model ---------------------
class DeltaRule:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.mse_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        
        for _ in range(self.epochs):
            net = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(net)
            error = y - y_pred
            grad = error * y_pred * (1 - y_pred)
            self.w += self.lr * np.dot(X.T, grad)
            self.b += self.lr * np.sum(grad)
            mse = np.mean(error**2)
            self.mse_history.append(mse)

    def predict(self, X):
        net = np.dot(X, self.w) + self.b
        return (self.sigmoid(net) >= 0.5).astype(int)

# --------------------- Train and Evaluate ---------------------
model = DeltaRule(lr=0.5, epochs=150)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100

print(f"Final Weights: {np.round(model.w, 3)}")
print(f"Bias: {round(model.b, 3)}")
print(f"Test Accuracy: {accuracy:.2f}%")

# --------------------- Plot MSE Convergence ---------------------
plt.plot(model.mse_history, color='red', lw=2)
plt.title("Delta Learning Rule – Mean Squared Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
