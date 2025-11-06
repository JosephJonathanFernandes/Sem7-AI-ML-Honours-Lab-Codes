# EBPA Classifier on Breast Cancer Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----- Activation Functions -----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ----- Load and Prepare Dataset -----
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Initialize Weights -----
np.random.seed(42)
n_input = X_train.shape[1]  # 30 features
n_hidden = 10
n_output = 1

w1 = np.random.randn(n_input, n_hidden)
b1 = np.random.randn(1, n_hidden)
w2 = np.random.randn(n_hidden, n_output)
b2 = np.random.randn(1, n_output)

lr = 0.01
epochs = 5000
errors = []

# ----- Training -----
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X_train, w1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w2) + b2
    output = sigmoid(final_input)

    # Error
    error = y_train - output
    errors.append(np.mean(np.abs(error)))

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_hidden = np.dot(d_output, w2.T) * sigmoid_derivative(hidden_output)

    # Update weights
    w2 += np.dot(hidden_output.T, d_output) * lr
    w1 += np.dot(X_train.T, d_hidden) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# ----- Testing -----
hidden_test = sigmoid(np.dot(X_test, w1) + b1)
output_test = sigmoid(np.dot(hidden_test, w2) + b2)
predictions = (output_test > 0.5).astype(int)

# Accuracy
accuracy = np.mean(predictions == y_test) * 100

# ----- Visualization -----
plt.plot(errors)
plt.title("Training Error using EBPA (Breast Cancer Dataset)")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.show()

print(f"Final Accuracy on Test Set: {accuracy:.2f}%")
