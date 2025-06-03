# train_models.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from .models import MNISTModel, CIFARModel
from utils import get_mnist_loaders, get_cifar_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4096

def train(model, train_loader, test_loader, epochs, lr, save_path):
    print("started training...")
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
    print(f"Best Accuracy: {best_acc:.4f}, Model saved to {save_path}")


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == '__main__':
    # Train MNIST
    os.makedirs("../models", exist_ok=True)
    mnist_train, mnist_test = get_mnist_loaders(BATCH_SIZE)
    mnist_model = MNISTModel()
    train(mnist_model, mnist_train, mnist_test, epochs=5, lr=1e-3, save_path='../models/mnist.pth')

    # Train CIFAR-10
    cifar_train, cifar_test = get_cifar_loaders(BATCH_SIZE)
    cifar_model = CIFARModel()
    train(cifar_model, cifar_train, cifar_test, epochs=40, lr=1e-3, save_path='../models/cifar.pth')