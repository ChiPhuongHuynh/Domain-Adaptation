"""
data/utils.py

Helper utilities for saving, loading, and preprocessing latent features
and dataset tensors.
"""

import torch
import os
from tqdm import tqdm


def normalize_tensor(x: torch.Tensor, method: str = "zscore"):
    """
    Normalize tensor along feature dimension.

    Args:
        x (torch.Tensor): Input tensor [N, D] or [N, C, H, W].
        method (str): One of {"zscore", "minmax"}.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    if method == "zscore":
        mean, std = x.mean(), x.std()
        return (x - mean) / (std + 1e-8)
    elif method == "minmax":
        minv, maxv = x.min(), x.max()
        return (x - minv) / (maxv - minv + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def save_latents(model, dataloader, device, save_path: str, key: str = "z"):
    """
    Extract and save latent embeddings for a dataset.
    This assumes `model(x)` returns either the latent or a dict containing it.

    Args:
        model: Encoder model.
        dataloader: DataLoader providing (x, y) pairs.
        device: torch device.
        save_path (str): Output file (.pt).
        key (str): If model returns dict, which key corresponds to latent.
    """
    model.eval()
    all_z, all_y = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Extracting latents → {save_path}"):
            x = x.to(device)
            out = model(x)
            if isinstance(out, dict):
                z = out[key]
            else:
                z = out
            all_z.append(z.cpu())
            all_y.append(y.cpu())

    Z = torch.cat(all_z)
    Y = torch.cat(all_y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"latents": Z, "labels": Y}, save_path)
    print(f"✅ Saved {Z.shape[0]} latent vectors to {save_path}")


def load_latents(path: str, device="cpu"):
    """
    Load latent embeddings and labels from .pt file.

    Returns:
        (torch.Tensor, torch.Tensor): latents, labels
    """
    data = torch.load(path, map_location=device)
    return data["latents"], data["labels"]


def concatenate_latents(paths, device="cpu"):
    """
    Load and concatenate multiple latent .pt files into a single tensor.
    Useful for merging train/test or domain splits.
    """
    zs, ys = [], []
    for p in paths:
        z, y = load_latents(p, device)
        zs.append(z)
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)
