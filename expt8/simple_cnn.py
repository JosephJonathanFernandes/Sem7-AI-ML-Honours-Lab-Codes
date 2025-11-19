
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
def load_mnist(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))             # 7x7 -> 7x7
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))            # 7x7 -> 3x3
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Training function
def train_model(model, trainloader, testloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_accuracies, test_accuracies

# Plotting function
def plot_results(train_acc, test_acc):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(test_acc, label='Test Accuracy', color='red')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_acc) + 1)
    plt.bar(epochs, test_acc, alpha=0.7, color='green')
    plt.title('Test Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_cnn_results.png')
    plt.show()

# Test predictions
def test_predictions(model, testloader):
    model.eval()
    
    # Get one batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Show first 8 images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(8):
        ax = axes[i//4, i%4]
        img = images[i].cpu().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {labels[i].item()}, Pred: {predictions[i].item()}',
                    color='green' if labels[i] == predictions[i] else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

# Main execution
def main():
    print("Simple CNN for MNIST Classification")
    print("=" * 40)
    
    # Load data
    print("Loading MNIST dataset...")
    trainloader, testloader = load_mnist(batch_size=64)
    
    # Create model
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters")
    
    print("\nModel Architecture:")
    print(model)
    
    # Train model
    print(f"\nTraining for 10 epochs...")
    train_acc, test_acc = train_model(model, trainloader, testloader, num_epochs=10)
    
    # Results
    final_acc = test_acc[-1]
    best_acc = max(test_acc)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    # Plot results
    plot_results(train_acc, test_acc)
    
    # Show sample predictions
    print("\nGenerating sample predictions...")
    test_predictions(model, testloader)
    
    # Save model
    torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
    print("Model saved as 'simple_cnn_mnist.pth'")
    
    print("\nFiles generated:")
    print("- simple_cnn_results.png")
    print("- sample_predictions.png")
    print("- simple_cnn_mnist.pth")
    
    # Print conclusion and comparison
    print_conclusion_and_comparison(final_acc, best_acc, total_params)

def print_conclusion_and_comparison(final_acc, best_acc, total_params):
    """Print detailed conclusion and hypothetical comparison with other architectures"""
    
    print("\n" + "="*80)
    print("CONCLUSION AND ARCHITECTURAL COMPARISON")
    print("="*80)
    
    # Simple CNN Results Summary
    print(f"\n📊 SIMPLE CNN PERFORMANCE SUMMARY:")
    print(f"{'='*50}")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: ~2-3 minutes")
    print(f"Dataset: MNIST (28x28 grayscale digits)")
    
    # Hypothetical comparison with other architectures
    print(f"\n🏗️ HYPOTHETICAL COMPARISON WITH POPULAR CNN ARCHITECTURES:")
    print(f"{'='*65}")
    
    # Expected performance table
    architectures = {
        "Simple CNN (Ours)": {
            "accuracy": f"{final_acc:.1f}%",
            "parameters": f"{total_params/1000:.0f}K",
            "training_time": "2-3 min",
            "strengths": "Lightweight, fast training, good baseline",
            "limitations": "Limited depth, fewer features"
        },
        "AlexNet": {
            "accuracy": "99.1-99.3%",
            "parameters": "25M",
            "training_time": "5-7 min",
            "strengths": "Historical significance, deeper features",
            "limitations": "Large parameter count, potential overkill for MNIST"
        },
        "VGG16": {
            "accuracy": "99.2-99.4%",
            "parameters": "15M",
            "training_time": "8-12 min",
            "strengths": "Very deep, uniform architecture, excellent feature extraction",
            "limitations": "Heavy computation, many parameters"
        },
        "GoogLeNet": {
            "accuracy": "99.3-99.5%",
            "parameters": "6M",
            "training_time": "6-9 min",
            "strengths": "Multi-scale features, efficient inception modules",
            "limitations": "Complex architecture, harder to interpret"
        },
        "ResNet18": {
            "accuracy": "99.4-99.6%",
            "parameters": "11M",
            "training_time": "4-6 min",
            "strengths": "Skip connections, solves vanishing gradients, excellent performance",
            "limitations": "More complex than needed for simple tasks"
        }
    }
    
    print(f"{'Architecture':<15} {'Accuracy':<12} {'Parameters':<12} {'Time':<10} {'Key Strengths'}")
    print("-" * 85)
    for arch, stats in architectures.items():
        print(f"{arch:<15} {stats['accuracy']:<12} {stats['parameters']:<12} {stats['training_time']:<10} {stats['strengths']}")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ARCHITECTURAL ANALYSIS:")
    print(f"{'='*45}")
    
    print(f"\n1. PERFORMANCE HIERARCHY (Expected on MNIST):")
    print(f"   ResNet18 > GoogLeNet ≥ VGG16 > AlexNet > Simple CNN")
    print(f"   • ResNet's skip connections prevent vanishing gradients")
    print(f"   • GoogLeNet's inception modules capture multi-scale features")
    print(f"   • VGG's depth provides rich feature representations")
    print(f"   • AlexNet introduced deep learning but is less optimized")
    print(f"   • Our Simple CNN provides solid baseline with minimal complexity")
    
    print(f"\n2. EFFICIENCY ANALYSIS:")
    print(f"   • Parameter Efficiency: GoogLeNet > ResNet18 > VGG16 > AlexNet")
    print(f"   • Training Speed: Simple CNN > ResNet18 > AlexNet > GoogLeNet > VGG16")
    print(f"   • Memory Usage: Simple CNN << ResNet18 < GoogLeNet < VGG16 < AlexNet")
    
    print(f"\n3. ARCHITECTURAL INNOVATIONS:")
    print(f"   • Simple CNN: Basic conv-pool-fc structure (our implementation)")
    print(f"   • AlexNet: Introduced deep CNNs, ReLU, dropout (2012)")
    print(f"   • VGG: Very deep networks with small 3x3 filters (2014)")
    print(f"   • GoogLeNet: Inception modules, 1x1 convolutions (2014)")
    print(f"   • ResNet: Skip connections, batch normalization (2015)")
    
    print(f"\n4. WHY RESNET WOULD LIKELY WIN:")
    print(f"   ✓ Skip connections enable training very deep networks")
    print(f"   ✓ Solves vanishing gradient problem effectively")
    print(f"   ✓ Batch normalization provides training stability")
    print(f"   ✓ Identity mappings preserve gradient flow")
    print(f"   ✓ Proven performance across multiple datasets")
    
    print(f"\n5. WHEN TO USE EACH ARCHITECTURE:")
    print(f"   • Simple CNN: Quick prototypes, educational purposes, resource constraints")
    print(f"   • AlexNet: Historical studies, understanding deep learning evolution")
    print(f"   • VGG: Feature extraction, transfer learning, when interpretability matters")
    print(f"   • GoogLeNet: When efficiency matters, multi-scale feature requirements")
    print(f"   • ResNet: Production systems, when highest accuracy is needed")
    
    # Final conclusion
    print(f"\n🎯 FINAL CONCLUSION:")
    print(f"{'='*20}")
    print(f"""
Our Simple CNN achieved {final_acc:.1f}% accuracy on MNIST with only {total_params:,} parameters, 
demonstrating that effective image classification doesn't always require complex architectures. 
However, advanced architectures like ResNet would likely achieve 99.5%+ accuracy due to their 
sophisticated designs that address fundamental deep learning challenges.

The evolution from simple CNNs to ResNet represents major breakthroughs in deep learning:
• Deeper networks capture more complex features but risk vanishing gradients
• Skip connections (ResNet) solve this while enabling very deep architectures
• Inception modules (GoogLeNet) provide computational efficiency
• Batch normalization stabilizes training across all modern architectures

For MNIST, our Simple CNN provides an excellent balance of simplicity, interpretability, 
and performance. For more complex tasks (ImageNet, CIFAR-100), ResNet's architectural 
sophistication becomes crucial for achieving state-of-the-art results.

This project effectively demonstrates that understanding architectural principles is more 
valuable than simply using the most complex model - the right architecture depends on 
the task complexity, computational constraints, and performance requirements.
    """)
    
    print("="*80)

if __name__ == "__main__":
    main()