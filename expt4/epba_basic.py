# EBPA Classifier on XOR Problem
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation and derivative
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x * (1 - x)

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.rand(2, 3)
b1 = np.random.rand(1, 3)
w2 = np.random.rand(3, 1)
b2 = np.random.rand(1, 1)

lr = 0.5  # learning rate
errors = []

# Training loop
for epoch in range(10000):
    # ----- Forward propagation -----
    hidden_input = np.dot(X, w1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w2) + b2
    output = sigmoid(final_input)
    
    # ----- Error calculation -----
    error = y - output
    errors.append(np.mean(np.abs(error)))
    
    # ----- Backward propagation -----
    d_output = error * sigmoid_derivative(output)
    d_hidden = np.dot(d_output, w2.T) * sigmoid_derivative(hidden_output)
    
    # ----- Weight updates -----
    w2 += np.dot(hidden_output.T, d_output) * lr
    w1 += np.dot(X.T, d_hidden) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# ----- Visualization -----
plt.plot(errors)
plt.title("Training Error using EBPA")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.show()

print("Final Output after training:\n", np.round(output, 2))
