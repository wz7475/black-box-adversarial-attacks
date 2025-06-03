import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ResultLogger:
    def __init__(self, output_dir, args):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
        with open(os.path.join(self.output_dir, 'log.csv'), 'w') as f:
            f.write("idx,true_label,pred_label,success,queries,iterations\n")

    def add_result(self, idx, true_label, pred_label, success, queries, iterations):
        with open(os.path.join(self.output_dir, 'log.csv'), 'a') as f:
            f.write(f"{idx},{true_label},{pred_label},{int(success)},{queries},{iterations}\n")


def get_mnist_loaders(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)
    train_set = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cifar_loaders(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.CIFAR10(root='data/cifar', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    train_set = datasets.CIFAR10(root='data/cifar', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
