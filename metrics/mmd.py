"""
metrics/mmd.py

Implements Maximum Mean Discrepancy (MMD) with RBF kernel.
Measures distribution distance between two sets of samples.
"""

import torch
import torch.nn.functional as F

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

def compute_pairwise_distances(x, y):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    return x_norm + y_norm - 2.0 * torch.mm(x, y.t())

def gaussian_kernel(x, y, sigmas=[0.1, 1, 5, 10]):
    D = compute_pairwise_distances(x, y)
    kernels = [torch.exp(-D / (2 * sigma ** 2)) for sigma in sigmas]
    return sum(kernels) / len(kernels)

@torch.no_grad()
def mmd_gaussian(x, y, sigmas=[0.1, 1, 5, 10]):
    K_xx = gaussian_kernel(x, x, sigmas)
    K_yy = gaussian_kernel(y, y, sigmas)
    K_xy = gaussian_kernel(x, y, sigmas)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd.item()
