import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------------- DEFINE NEURAL NETWORK -------------------
class LogicGateNN(nn.Module):
    def __init__(self, input_size, hidden_size=4):
        super(LogicGateNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# ------------------- TRAINING FUNCTION -------------------
def train_gate(X, y, gate_name, epochs=1000, lr=0.1):
    print(f"\nTraining {gate_name} Gate")

    # Convert to tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Model
    model = LogicGateNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        predictions = (model(X_train) > 0.5).int()
        accuracy = (predictions == y_train.int()).float().mean()
        print(f"{gate_name} Gate Accuracy: {accuracy*100:.2f}%")
        print(f"Inputs:\n{X}\nPredictions:\n{predictions.numpy()}\nExpected:\n{y}\n")
    
    return model

# ------------------- VISUALIZATION FUNCTION -------------------
def plot_gate(X, y, model, gate_name):
    X = np.array(X)
    y = np.array(y)
    if X.shape[1] == 2:  # 2-input gates
        x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
        y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        with torch.no_grad():
            Z = (model(grid_tensor) > 0.5).int().numpy()
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:,0], X[:,1], c=y[:,0], s=100, edgecolors='k', cmap=plt.cm.RdYlBu)
        plt.title(f"{gate_name} Gate Decision Boundary")
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.show()
    else:  # Single-input NOT gate
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = (model(X_tensor) > 0.5).int().numpy()
        plt.bar([0,1], predictions[:,0], color='skyblue', alpha=0.7)
        plt.scatter([0,1], y[:,0], color='red', s=100, label='Expected')
        plt.xticks([0,1])
        plt.ylim(-0.1, 1.1)
        plt.title(f"{gate_name} Gate Predictions")
        plt.ylabel("Output")
        plt.legend()
        plt.show()


# ------------------- TRAIN AND VISUALIZE ALL LOGIC GATES -------------------

# AND gate
X_and = [[0,0],[0,1],[1,0],[1,1]]
y_and = [[0],[0],[0],[1]]
model_and = train_gate(X_and, y_and, "AND")
plot_gate(X_and, y_and, model_and, "AND")

# OR gate
X_or = [[0,0],[0,1],[1,0],[1,1]]
y_or = [[0],[1],[1],[1]]
model_or = train_gate(X_or, y_or, "OR")
plot_gate(X_or, y_or, model_or, "OR")

# NOT gate (single input)
X_not = [[0],[1]]
y_not = [[1],[0]]
model_not = train_gate(X_not, y_not, "NOT")
plot_gate(X_not, y_not, model_not, "NOT")

# XOR gate (non-linear, requires hidden layer)
X_xor = [[0,0],[0,1],[1,0],[1,1]]
y_xor = [[0],[1],[1],[0]]
model_xor = train_gate(X_xor, y_xor, "XOR")
plot_gate(X_xor, y_xor, model_xor, "XOR")
