# Imports and MNIST data
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ----------------------------
# 1️⃣ Classical Autoencoder
# ----------------------------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

# ----------------------------
# 2️⃣ Denoising Autoencoder
# ----------------------------
class DAE(AE):
    def forward(self, x):
        noise = torch.randn_like(x) * 0.4
        x_noisy = torch.clamp(x + noise, 0., 1.)
        return self.decoder(self.encoder(x_noisy))

# ----------------------------
# 3️⃣ Variational Autoencoder
# ----------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc_mu = nn.Linear(512, 2)
        self.fc_logvar = nn.Linear(512, 2)
        self.fc2 = nn.Linear(2, 512)
        self.fc3 = nn.Linear(512, 784)
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------------------------
# 4️⃣ Loss Functions
# ----------------------------
def mse_loss(recon, x): return ((recon - x)**2).mean()
def vae_loss(recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)

# ----------------------------
# 5️⃣ Training (compare results)
# ----------------------------
def train_model(model, optimizer, loader, is_vae=False):
    model.train()
    for epoch in range(3):
        total_loss = 0
        for x, _ in loader:
            if is_vae:
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar)
            else:
                recon = model(x)
                loss = mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

# Train and compare
ae, dae, vae = AE(), DAE(), VAE()
opt1, opt2, opt3 = optim.Adam(ae.parameters(), lr=1e-3), optim.Adam(dae.parameters(), lr=1e-3), optim.Adam(vae.parameters(), lr=1e-3)

print("\nTraining Classical AE:")
train_model(ae, opt1, train_loader)

print("\nTraining Denoising AE:")
train_model(dae, opt2, train_loader)

print("\nTraining Variational AE:")
train_model(vae, opt3, train_loader, is_vae=True)
