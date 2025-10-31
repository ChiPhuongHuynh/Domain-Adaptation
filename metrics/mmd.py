"""
metrics/mmd.py

Implements Maximum Mean Discrepancy (MMD) with RBF kernel.
Measures distribution distance between two sets of samples.
"""

import torch

def _rbf_kernel(x, y, sigma):
    """Compute RBF (Gaussian) kernel matrix."""
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between X and Y using RBF kernel.

    Args:
        X: [N, D] tensor
        Y: [M, D] tensor
        sigma: kernel bandwidth

    Returns:
        Scalar tensor (larger = more dissimilar)
    """
    X = X.float()
    Y = Y.float()

    Kxx = _rbf_kernel(X, X, sigma).mean()
    Kyy = _rbf_kernel(Y, Y, sigma).mean()
    Kxy = _rbf_kernel(X, Y, sigma).mean()

    return Kxx + Kyy - 2 * Kxy
