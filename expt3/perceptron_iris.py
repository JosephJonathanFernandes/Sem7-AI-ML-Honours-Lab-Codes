"""
Title: Perceptron Learning Algorithm using Iris Dataset
Author: Joseph Jonathan Fernandes
Description:
Implements a Perceptron to classify two Iris flower species (Setosa vs Versicolor)
using sepal length and sepal width features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------- Load and Prepare Dataset ---------------------
iris = load_iris()
X = iris.data[:100, :2]   # Sepal length, Sepal width
y = iris.target[:100]     # 0 = Setosa, 1 = Versicolor

# Standardize for faster convergence
X = StandardScaler().fit_transform(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Perceptron Model ---------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.errors = []

        for _ in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.w) + self.b
                y_pred = self.activation(z)
                update = self.lr * (target - y_pred)
                self.w += update * xi
                self.b += update
                total_error += int(update != 0)
            self.errors.append(total_error)

    def predict(self, X):
        return self.activation(np.dot(X, self.w) + self.b)

# --------------------- Train and Evaluate ---------------------
model = Perceptron(lr=0.1, epochs=30)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100

print(f"Final Weights: {np.round(model.w, 3)}")
print(f"Bias: {round(model.b, 3)}")
print(f"Test Accuracy: {accuracy:.2f}%")

# --------------------- Plot Error Convergence ---------------------
plt.plot(model.errors, marker='o')
plt.title("Perceptron Learning – Misclassifications per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.grid(True)
plt.show()
