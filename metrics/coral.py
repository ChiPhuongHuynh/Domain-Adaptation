"""
metrics/coral.py

Implements CORrelation ALignment (CORAL) distance between two feature sets.
Reference: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)
"""

import torch

def coral_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL distance between two feature matrices X and Y.

    Args:
        X: [N, D] tensor (e.g. features from domain A)
        Y: [M, D] tensor (e.g. features from domain B)

    Returns:
        Scalar tensor representing squared Frobenius norm distance
        between covariance matrices + mean difference.
    """
    # Ensure float type
    X = X.float()
    Y = Y.float()

    # Means
    mean_X = X.mean(0, keepdim=True)
    mean_Y = Y.mean(0, keepdim=True)

    # Centered
    X_c = X - mean_X
    Y_c = Y - mean_Y

    # Covariances
    cov_X = (X_c.T @ X_c) / (X.shape[0] - 1)
    cov_Y = (Y_c.T @ Y_c) / (Y.shape[0] - 1)

    # Mean difference term
    mean_diff = torch.sum((mean_X - mean_Y) ** 2)

    # Covariance Frobenius distance term
    cov_diff = torch.sum((cov_X - cov_Y) ** 2)

    return mean_diff + cov_diff
