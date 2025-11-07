import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ----------------------------
# 1️⃣ Data preparation
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ----------------------------
# 2️⃣ Variational Autoencoder model
# ----------------------------
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc_mu = nn.Linear(512, 2)       # mean
        self.fc_logvar = nn.Linear(512, 2)   # log variance
        # Decoder
        self.fc2 = nn.Linear(2, 512)
        self.fc3 = nn.Linear(512, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------------------------
# 3️⃣ Loss Function
# ----------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------------------------
# 4️⃣ Training Loop
# ----------------------------
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for x, _ in train_loader:
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader.dataset):.4f}")

print("✅ Variational Autoencoder training completed successfully!")
