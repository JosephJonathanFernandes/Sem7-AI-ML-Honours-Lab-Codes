import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation + derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Multi-layer perceptron class
class MLP:
    def __init__(self, input_size, hidden_size=2, lr=0.1, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros((1, 1))
        self.loss_history = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        for _ in range(self.epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = sigmoid(z2)

            # Compute loss (mean squared error)
            loss = np.mean((y - a2) ** 2)
            self.loss_history.append(loss)

            # Backpropagation
            error = y - a2
            d_a2 = error * sigmoid_derivative(a2)

            error_hidden = d_a2.dot(self.W2.T)
            d_a1 = error_hidden * sigmoid_derivative(a1)

            # Update weights
            self.W2 += a1.T.dot(d_a2) * self.lr
            self.b2 += np.sum(d_a2, axis=0, keepdims=True) * self.lr
            self.W1 += X.T.dot(d_a1) * self.lr
            self.b1 += np.sum(d_a1, axis=0, keepdims=True) * self.lr

        # After training
        predictions = self.predict(X)
        print(f"{self.gate_name} Gate:")
        print(f"Inputs:\n{X}\nPredictions:\n{predictions.flatten()}\nExpected:\n{y.flatten()}\n")

    def predict(self, X):
        X = np.array(X)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        return np.round(a2)

    def plot_loss(self):
        plt.plot(self.loss_history, label=self.gate_name)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve - {self.gate_name}")
        plt.legend()
        plt.show()

    def set_gate_name(self, name):
        self.gate_name = name

    # Visualization for 2-input gates or bar plot for NOT
    def plot_gate(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1,1)

        if X.shape[1] == 2:  # 2-input gates
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = self.predict(grid_points).reshape(xx.shape)

            plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
            plt.scatter(X[:,0], X[:,1], c=y[:,0], s=100, edgecolors='k', cmap=plt.cm.RdYlBu)
            plt.title(f"{self.gate_name} Gate Decision Boundary")
            plt.xlabel("Input 1")
            plt.ylabel("Input 2")
            plt.show()
        else:  # NOT gate (single input)
            predictions = self.predict(X)
            plt.bar([0,1], predictions.flatten(), color='skyblue', alpha=0.7)
            plt.scatter([0,1], y.flatten(), color='red', s=100, label='Expected')
            plt.xticks([0,1])
            plt.ylim(-0.1,1.1)
            plt.title(f"{self.gate_name} Gate Predictions")
            plt.ylabel("Output")
            plt.legend()
            plt.show()


# ------------------- TRAIN AND VISUALIZE ALL LOGIC GATES -------------------

gates = {
    "AND": {"X": [[0,0],[0,1],[1,0],[1,1]], "y": [0,0,0,1]},
    "OR":  {"X": [[0,0],[0,1],[1,0],[1,1]], "y": [0,1,1,1]},
    "XOR": {"X": [[0,0],[0,1],[1,0],[1,1]], "y": [0,1,1,0]},
    "NOT": {"X": [[0],[1]], "y": [1,0]}
}

for gate_name, data in gates.items():
    input_size = len(data["X"][0])
    hidden_size = 4 if gate_name == "XOR" else 2  # XOR needs more neurons
    mlp = MLP(input_size=input_size, hidden_size=hidden_size, epochs=10000, lr=0.1)
    mlp.set_gate_name(gate_name)
    mlp.train(data["X"], data["y"])
    mlp.plot_loss()
    mlp.plot_gate(data["X"], data["y"])


