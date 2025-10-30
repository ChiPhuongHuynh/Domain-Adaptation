
"""
data/loaders.py

Dataset loading utilities for MNIST and USPS.
All datasets are normalized and resized to 28×28 grayscale for consistency.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _default_transform():
    """Return a standard transform pipeline for grayscale 28×28 digits."""
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])


def get_mnist(batch_size=128, train=True, num_workers=2):
    """
    Load MNIST dataset with standard preprocessing.
    Args:
        batch_size (int): Batch size for DataLoader.
        train (bool): Whether to load training or test split.
    Returns:
        DataLoader: MNIST dataloader.
    """
    transform = _default_transform()
    dataset = datasets.MNIST(
        root="data/mnist",
        train=train,
        download=True,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def get_usps(batch_size=128, train=True, num_workers=2):
    """
    Load USPS dataset with standard preprocessing.
    Args:
        batch_size (int): Batch size for DataLoader.
        train (bool): Whether to load training or test split.
    Returns:
        DataLoader: USPS dataloader resized to 28×28.
    """
    transform = _default_transform()
    dataset = datasets.USPS(
        root="data/usps",
        train=train,
        download=True,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def get_dataloader(dataset_name: str, batch_size=128, train=True, num_workers=2):
    """
    Unified loader interface for MNIST and USPS.
    Example:
        train_loader = get_dataloader("mnist", 128, train=True)
        test_loader = get_dataloader("usps", 128, train=False)
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return get_mnist(batch_size, train, num_workers)
    elif dataset_name == "usps":
        return get_usps(batch_size, train, num_workers)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
