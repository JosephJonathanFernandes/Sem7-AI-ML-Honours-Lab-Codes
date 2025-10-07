import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_data_loaders(batch_size=128, data_dir='./data'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def eval_model(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def plot_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = len(history['train_loss'])
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='train')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='train')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'))
    plt.close()


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 with a simple CNN')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    trainloader, testloader = get_data_loaders(batch_size=args.batch_size, data_dir=args.data_dir)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, trainloader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, device, testloader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, is_best, args.checkpoint_dir)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}%, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes. Best val acc: {best_acc:.2f}%")

    plot_curves(history, out_dir=args.checkpoint_dir)

    # Save final history
    np.savez(os.path.join(args.checkpoint_dir, 'history.npz'), **history)


if __name__ == '__main__':
    main()
