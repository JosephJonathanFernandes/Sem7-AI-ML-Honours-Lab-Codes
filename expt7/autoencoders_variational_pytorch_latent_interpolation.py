import numpy as np
import matplotlib.pyplot as plt

# Take two random latent points in 2D
z1 = torch.tensor([-2.0, -2.0])
z2 = torch.tensor([2.0, 2.0])

# Linearly interpolate between them
interpolations = [z1 + (z2 - z1) * i/9 for i in range(10)]

# Simple decoder for visualization (example)
decoder = nn.Sequential(nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid())

with torch.no_grad():
    images = [decoder(z).view(28, 28) for z in interpolations]

plt.figure(figsize=(10, 2))
for i, img in enumerate(images):
    plt.subplot(1, 10, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle("VAE Latent Interpolation", fontsize=14)
plt.show()
