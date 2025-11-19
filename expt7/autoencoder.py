"""
Basic Autoencoder Implementation for Data Compression and Reconstruction
Experiment 7 - Neural Networks and Deep Learning

This implementation demonstrates:
1. Basic autoencoder architecture with encoder-decoder structure
2. Training on MNIST dataset
3. Visualization of compression and reconstruction
4. Analysis of learned representations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BasicAutoencoder(nn.Module):
    """
    Basic Autoencoder Architecture
    
    Encoder: Input (784) -> Hidden (128) -> Compressed (32)
    Decoder: Compressed (32) -> Hidden (128) -> Output (784)
    """
    
    def __init__(self, input_dim=784, hidden_dim=128, encoding_dim=32):
        super(BasicAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        """Get only the encoded representation"""
        return self.encoder(x)
    
    def decode(self, encoded):
        """Get only the decoded representation"""
        return self.decoder(encoded)

def prepare_data(batch_size=128):
    """
    Prepare MNIST dataset for autoencoder training
    """
    # Transform: Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def train_autoencoder(model, train_loader, num_epochs=20, learning_rate=1e-3):
    """
    Train the autoencoder model
    """
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Flatten the images
            data = data.view(data.size(0), -1).to(device)
            
            # Forward pass
            reconstructed, encoded = model(data)
            
            # Calculate loss (reconstruction error)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
    
    return train_losses

def evaluate_autoencoder(model, test_loader):
    """
    Evaluate the autoencoder on test data
    """
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.6f}')
    return avg_test_loss

def visualize_results(model, test_loader, num_images=10):
    """
    Visualize original vs reconstructed images
    """
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select first num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Flatten and move to device
    flattened_images = images.view(images.size(0), -1).to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, encoded = model(flattened_images)
    
    # Move back to CPU and reshape
    original_images = images.cpu().numpy()
    reconstructed_images = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    encoded_representations = encoded.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        # Original images
        axes[0, i].imshow(original_images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {labels[i].item()}')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
        
        # Encoded representations (show as bar plot)
        axes[2, i].bar(range(len(encoded_representations[i])), encoded_representations[i])
        axes[2, i].set_title(f'Encoded\n(32 features)')
        axes[2, i].set_ylim(0, max(encoded_representations[i]) * 1.1)
    
    plt.tight_layout()
    plt.savefig('autoencoder_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return original_images, reconstructed_images, encoded_representations

def visualize_latent_space(model, test_loader, num_samples=1000):
    """
    Visualize the learned latent space using t-SNE
    """
    model.eval()
    
    encoded_samples = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            if len(encoded_samples) * test_loader.batch_size >= num_samples:
                break
                
            data = data.view(data.size(0), -1).to(device)
            encoded = model.encode(data)
            
            encoded_samples.append(encoded.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # Concatenate all samples
    encoded_samples = np.concatenate(encoded_samples, axis=0)[:num_samples]
    labels_array = np.concatenate(labels_list, axis=0)[:num_samples]
    
    # Apply t-SNE for visualization
    print("Applying t-SNE to latent representations...")
    tsne = TSNE(n_components=2, random_state=42)
    encoded_2d = tsne.fit_transform(encoded_samples)
    
    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], 
                         c=labels_array, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Latent Space\n(Colored by MNIST digit labels)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('latent_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_training_loss(train_losses):
    """
    Plot training loss over epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_compression_ratio(model):
    """
    Analyze the compression achieved by the autoencoder
    """
    input_dim = 784  # 28x28 pixels
    encoding_dim = 32  # Compressed representation
    
    compression_ratio = input_dim / encoding_dim
    compression_percentage = (1 - encoding_dim / input_dim) * 100
    
    print(f"\n--- Compression Analysis ---")
    print(f"Original dimension: {input_dim}")
    print(f"Compressed dimension: {encoding_dim}")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Data reduction: {compression_percentage:.1f}%")
    
    return compression_ratio, compression_percentage

def main():
    """
    Main function to run the autoencoder experiment
    """
    print("=== Basic Autoencoder for Data Compression and Reconstruction ===\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    print("1. Preparing MNIST dataset...")
    train_loader, test_loader = prepare_data(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print("2. Creating autoencoder model...")
    model = BasicAutoencoder(input_dim=784, hidden_dim=128, encoding_dim=32).to(device)
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Analyze compression
    analyze_compression_ratio(model)
    
    # Train the model
    print("\n3. Training the autoencoder...")
    train_losses = train_autoencoder(model, train_loader, num_epochs=20, learning_rate=1e-3)
    
    # Plot training loss
    print("\n4. Plotting training progress...")
    plot_training_loss(train_losses)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_loss = evaluate_autoencoder(model, test_loader)
    
    # Visualize results
    print("\n6. Visualizing reconstruction results...")
    original, reconstructed, encoded = visualize_results(model, test_loader, num_images=10)
    
    # Visualize latent space
    print("\n7. Visualizing latent space...")
    visualize_latent_space(model, test_loader, num_samples=1000)
    
    # Save the model
    print("\n8. Saving trained model...")
    torch.save(model.state_dict(), 'basic_autoencoder.pth')
    print("Model saved as 'basic_autoencoder.pth'")
    
    print("\n=== Experiment Complete ===")
    print(f"Final test loss: {test_loss:.6f}")
    print("Generated files:")
    print("- autoencoder_results.png (original vs reconstructed)")
    print("- latent_space_visualization.png (t-SNE of latent space)")
    print("- training_loss.png (loss over epochs)")
    print("- basic_autoencoder.pth (trained model)")

if __name__ == "__main__":
    main()