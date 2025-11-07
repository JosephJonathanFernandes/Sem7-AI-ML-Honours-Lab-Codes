import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

# ----------------------------
# 1️⃣ Load Data
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ----------------------------
# 2️⃣ Pretrained / Trained Models (simplified)
# ----------------------------
class EncoderAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x): return self.encoder(x)

class EncoderDAE(EncoderAE):
    def forward(self, x):
        noise = torch.randn_like(x) * 0.4
        return self.encoder(torch.clamp(x + noise, 0., 1.))

class EncoderVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc_mu = nn.Linear(512, 2)
        self.fc_logvar = nn.Linear(512, 2)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, mu, logvar

# Instantiate (normally load trained weights)
ae = EncoderAE()
dae = EncoderDAE()
vae = EncoderVAE()

# ----------------------------
# 3️⃣ Extract Latent Representations
# ----------------------------
def get_latents(model, loader, vae=False):
    all_latents, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            if vae:
                z, _, _ = model(x)
            else:
                z = model(x)
            all_latents.append(z)
            all_labels.append(y)
    return torch.cat(all_latents), torch.cat(all_labels)

lat_ae, labels = get_latents(ae, test_loader)
lat_dae, _ = get_latents(dae, test_loader)
lat_vae, _ = get_latents(vae, test_loader, vae=True)

# ----------------------------
# 4️⃣ Dimensionality Reduction (PCA to 2D if needed)
# ----------------------------
def reduce_to_2d(latent):
    if latent.shape[1] > 2:
        latent = PCA(n_components=2).fit_transform(latent.numpy())
    else:
        latent = latent.numpy()
    return latent

ae_2d = reduce_to_2d(lat_ae)
dae_2d = reduce_to_2d(lat_dae)
vae_2d = reduce_to_2d(lat_vae)

# ----------------------------
# 5️⃣ Visualization
# ----------------------------
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.scatter(ae_2d[:, 0], ae_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Classical Autoencoder Latent Space")

plt.subplot(1, 3, 2)
plt.scatter(dae_2d[:, 0], dae_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Denoising Autoencoder Latent Space")

plt.subplot(1, 3, 3)
plt.scatter(vae_2d[:, 0], vae_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.title("Variational Autoencoder Latent Space")

plt.suptitle("Latent Space Visualization of Autoencoders", fontsize=15)
plt.show()
